#!/usr/bin/env python3
"""
MedlinePlus Article Scraper
Scrapes articles from medlineplus.gov and saves to JSON and TXT files
This version can work with pre-fetched HTML content or fetch directly from URL
"""

import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
import re
import sys


def parse_article_html(html_content, url):
    """
    Parse MedlinePlus article HTML and extract structured content

    Args:
        html_content: HTML string or file content
        url: Original URL of the article

    Returns:
        dict: Structured article data
    """
    # Parse HTML
    soup = BeautifulSoup(html_content, 'html.parser')

    # Extract title
    title = soup.find('h1')
    title_text = title.get_text(strip=True) if title else "No title found"

    # Clean the title (remove "MedlinePlus Medical Encyclopedia" suffix)
    title_text = re.sub(r'\s*:\s*MedlinePlus.*$', '', title_text)

    # Extract main content sections
    content_sections = []
    current_section = None

    # Find the main content area - look for the actual article content
    # Skip navigation, headers, footers
    skip_patterns = [
        'skip navigation', 'medlineplus', 'you are here',
        'health topics', 'url of this page', 'browse the encyclopedia',
        'national library of medicine', 'subscribe to rss',
        'a.d.a.m.', 'urac', 'privacy policy', 'official website',
        'connect with nlm', 'return to top', 'customer support',
        "here's how you know", 'hereâ€™s how you know',
        'secure .gov websites', 'https', 'locked padlock',
        'drugs & supplements', 'genetics', 'medical tests',
        'medical encyclopedia', "what's new", 'site map',
        'nlm web policies', 'copyright', 'accessibility',
        'guidelines for links', 'viewers & players',
        'vulnerability disclosure', 'for developers',
        'health content provider', 'review date'
    ]

    # Additional patterns to skip list items that are just navigation
    nav_list_items = [
        'drugs & supplements', 'genetics', 'medical tests',
        'medical encyclopedia', "what's new", 'site map'
    ]

    # Process all paragraphs and headings
    for element in soup.find_all(['h2', 'h3', 'p', 'ul', 'li']):
        text = element.get_text(strip=True)

        # Skip empty or irrelevant content
        if not text or len(text) < 3:
            continue

        # Skip navigation and footer content
        text_lower = text.lower()
        if any(skip in text_lower for skip in skip_patterns):
            continue

        # Skip very long elements (likely navigation or footer)
        if len(text) > 1000:
            continue

        # Skip list items that are just single words or short navigation items
        if element.name == 'li' and len(text) < 20:
            if any(nav_item in text_lower for nav_item in nav_list_items):
                continue

        tag_name = element.name

        if tag_name in ['h2', 'h3']:
            # Check if this is a meaningful section heading
            # Skip review date headings and other meta headings
            if 'review date' in text_lower:
                continue
            if text not in ['References', 'Related MedlinePlus Health Topics']:
                # Start new section
                current_section = {
                    'heading': text,
                    'heading_level': tag_name,
                    'content': []
                }
                content_sections.append(current_section)
        elif tag_name == 'p' and current_section is not None:
            # Add paragraph to current section
            if text and len(text) > 10:  # Skip very short paragraphs
                current_section['content'].append(text)
        elif tag_name == 'p' and current_section is None:
            # Introduction paragraph before any heading
            if not content_sections or content_sections[0]['heading'] != 'Introduction':
                content_sections.insert(0, {
                    'heading': 'Introduction',
                    'heading_level': 'h2',
                    'content': []
                })
                current_section = content_sections[0]
            if text and len(text) > 10:
                current_section['content'].append(text)
        elif tag_name == 'li' and current_section is not None:
            # Add list items to current section
            if text and len(text) > 5:
                current_section['content'].append(f"• {text}")

    # Clean up the Introduction section - remove navigation items
    if content_sections and content_sections[0]['heading'] == 'Introduction':
        intro_section = content_sections[0]
        cleaned_content = []
        for item in intro_section['content']:
            item_lower = item.lower()
            # Skip if it's a navigation item
            if any(skip in item_lower for skip in skip_patterns):
                continue
            # Skip if it starts with a bullet and is short (likely navigation)
            if item.startswith('•') and len(item) < 30:
                continue
            # Keep substantial content
            if len(item) > 30:
                cleaned_content.append(item)

        intro_section['content'] = cleaned_content

        # If introduction is empty after cleaning, remove it
        if not cleaned_content:
            content_sections.pop(0)

    # Extract references
    references = []
    ref_heading = soup.find('h2', string=re.compile(r'^\s*References\s*$', re.I))
    if ref_heading:
        # Find all paragraphs after the references heading
        next_elem = ref_heading.find_next_sibling()
        while next_elem and next_elem.name != 'h2':
            if next_elem.name == 'p':
                ref_text = next_elem.get_text(strip=True)
                if ref_text and len(ref_text) > 10:
                    # Skip non-reference content
                    if not any(skip in ref_text.lower() for skip in skip_patterns):
                        references.append(ref_text)
            next_elem = next_elem.find_next_sibling()

    # Extract review date
    review_date = None
    review_heading = soup.find('h2', string=re.compile(r'^\s*Review Date\s*', re.I))
    if review_heading:
        date_elem = review_heading.find_next_sibling('p')
        if date_elem:
            review_date = date_elem.get_text(strip=True)

    # Extract related topics
    related_topics = []
    related_heading = soup.find('h2', string=re.compile(r'Related.*Health Topics', re.I))
    if related_heading:
        next_elem = related_heading.find_next_sibling()
        if next_elem and next_elem.name == 'ul':
            for link in next_elem.find_all('a'):
                topic_title = link.get_text(strip=True)
                topic_url = link.get('href', '')
                if topic_title:
                    # Make relative URLs absolute
                    if topic_url.startswith('http'):
                        full_url = topic_url
                    elif topic_url.startswith('/'):
                        full_url = f"https://medlineplus.gov{topic_url}"
                    else:
                        full_url = topic_url

                    related_topics.append({
                        'title': topic_title,
                        'url': full_url
                    })

    # Build structured data
    article_data = {
        'url': url,
        'title': title_text,
        'scraped_at': datetime.now().isoformat(),
        'content_sections': content_sections,
        'references': references,
        'review_date': review_date,
        'related_topics': related_topics
    }

    return article_data


def scrape_medlineplus_article(url):
    """
    Scrape a MedlinePlus article from URL

    Args:
        url: URL of the MedlinePlus article

    Returns:
        dict: Structured article data
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    return parse_article_html(response.content, url)


def save_to_json(data, filename):
    """Save data to JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved to {filename}")


def save_to_txt(data, filename):
    """Save data to TXT file in readable format"""
    with open(filename, 'w', encoding='utf-8') as f:
        # Write title
        f.write("=" * 80 + "\n")
        f.write(data['title'].upper() + "\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Source: {data['url']}\n")
        f.write(f"Scraped: {data['scraped_at']}\n")
        if data.get('review_date'):
            f.write(f"Review Date: {data['review_date']}\n")
        f.write("\n" + "-" * 80 + "\n\n")

        # Write content sections
        for section in data['content_sections']:
            # Write heading
            heading = section['heading'].upper()
            f.write(heading + "\n")
            f.write("=" * len(heading) + "\n\n")

            # Write content
            for paragraph in section['content']:
                f.write(paragraph + "\n\n")

            f.write("\n")

        # Write references
        if data['references']:
            f.write("\n" + "=" * 80 + "\n")
            f.write("REFERENCES\n")
            f.write("=" * 80 + "\n\n")
            for i, ref in enumerate(data['references'], 1):
                f.write(f"{i}. {ref}\n\n")

        # Write related topics
        if data['related_topics']:
            f.write("\n" + "=" * 80 + "\n")
            f.write("RELATED HEALTH TOPICS\n")
            f.write("=" * 80 + "\n\n")
            for topic in data['related_topics']:
                f.write(f"• {topic['title']}\n")
                f.write(f"  {topic['url']}\n\n")

    print(f"✓ Saved to {filename}")


def main():
    """Main function"""
    url = "https://medlineplus.gov/ency/patientinstructions/000921.htm"

    print(f"Scraping article from: {url}")
    print("-" * 80)

    try:
        # Scrape the article
        article_data = scrape_medlineplus_article(url)

        print(f"\nArticle Title: {article_data['title']}")
        print(f"Sections found: {len(article_data['content_sections'])}")
        print(f"References found: {len(article_data['references'])}")
        print(f"Related topics: {len(article_data['related_topics'])}")
        print("\n" + "-" * 80)

        # Save to files
        print("\nSaving files...")
        save_to_json(article_data, 'article.json')
        save_to_txt(article_data, 'article.txt')

        print("\n✓ Scraping completed successfully!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()