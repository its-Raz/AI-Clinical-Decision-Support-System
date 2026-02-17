#!/usr/bin/env python3
"""
MedlinePlus Bulk Encyclopedia Scraper
Scrapes all articles from the MedlinePlus encyclopedia
Saves to organized JSON and TXT folders
"""

import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
import re
import sys
import os
import time
from pathlib import Path
from urllib.parse import urljoin


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


def get_article_links_from_index(letter):
    """
    Get all article links from a letter index page

    Args:
        letter: Letter of the alphabet (A-Z) or '0-9'

    Returns:
        list: List of article URLs
    """
    index_url = f"https://medlineplus.gov/ency/encyclopedia_{letter}.htm"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    try:
        print(f"  Fetching index: {index_url}")
        response = requests.get(index_url, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all article links - they're typically in a list
        article_links = []

        # Look for links in the main content area
        for link in soup.find_all('a', href=True):
            href = link.get('href')

            # Article links can have various patterns:
            # - /ency/article/XXXXXX.htm
            # - /ency/patientinstructions/XXXXXX.htm
            # - article/XXXXXX.htm (relative)
            # - patientinstructions/XXXXXX.htm (relative)

            # Check if this is an encyclopedia article link
            is_article = False

            # Pattern 1: Full path with /ency/
            if '/ency/article/' in href or '/ency/patientinstructions/' in href:
                is_article = True

            # Pattern 2: Relative path
            elif href.startswith('article/') or href.startswith('patientinstructions/'):
                is_article = True
                # Make it a full /ency/ path
                href = f"/ency/{href}"

            # Pattern 3: Just the htm file
            elif re.match(r'^\d{6}\.htm$', href):
                is_article = True
                href = f"/ency/article/{href}"

            if is_article:
                # Convert to absolute URL
                if href.startswith('http'):
                    full_url = href
                else:
                    full_url = urljoin('https://medlineplus.gov', href)
                article_links.append(full_url)

        # Remove duplicates while preserving order
        seen = set()
        unique_links = []
        for link in article_links:
            if link not in seen:
                seen.add(link)
                unique_links.append(link)

        print(f"  Found {len(unique_links)} unique article links")
        return unique_links

    except Exception as e:
        print(f"  ✗ Error fetching index for letter {letter}: {e}")
        import traceback
        traceback.print_exc()
        return []


def scrape_article(url):
    """
    Scrape a single article

    Args:
        url: Article URL

    Returns:
        dict: Article data or None if error
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return parse_article_html(response.content, url)
    except Exception as e:
        print(f"  ✗ Error scraping {url}: {e}")
        return None


def sanitize_filename(title):
    """
    Convert title to safe filename

    Args:
        title: Article title

    Returns:
        str: Safe filename
    """
    # Remove invalid characters
    safe_title = re.sub(r'[<>:"/\\|?*]', '', title)
    # Replace spaces with underscores
    safe_title = safe_title.replace(' ', '_')
    # Limit length
    safe_title = safe_title[:100]
    return safe_title


def save_to_json(data, filepath):
    """Save data to JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_to_txt(data, filepath):
    """Save data to TXT file in readable format"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
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


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Scrape MedlinePlus Encyclopedia')
    parser.add_argument('--letters', type=str, help='Specific letters to scrape (e.g., "ABC" or "A-C")', default=None)
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--url-file', type=str, help='File containing list of URLs to scrape (one per line)',
                        default=None)
    parser.add_argument('--max-per-letter', type=int, help='Maximum articles per letter (for testing)', default=None)

    args = parser.parse_args()

    # Setup directories
    base_dir = Path('data_pipeline/data')
    json_dir = base_dir / 'json'
    txt_dir = base_dir / 'txt'

    json_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MedlinePlus Encyclopedia Bulk Scraper")
    print("=" * 80)
    print(f"\nOutput directories:")
    print(f"  JSON: {json_dir}")
    print(f"  TXT:  {txt_dir}")
    print("\n" + "=" * 80 + "\n")

    total_articles = 0
    successful_scrapes = 0
    failed_scrapes = 0

    # Mode 1: Scrape from URL file
    if args.url_file:
        print(f"Reading URLs from: {args.url_file}")
        try:
            with open(args.url_file, 'r') as f:
                article_links = [line.strip() for line in f if line.strip() and line.strip().startswith('http')]

            print(f"Found {len(article_links)} URLs in file")
            total_articles = len(article_links)

            for i, url in enumerate(article_links, 1):
                print(f"\nArticle {i}/{len(article_links)}: ", end='')

                article_data = scrape_article(url)

                if article_data is None:
                    failed_scrapes += 1
                    continue

                print(f"✓ {article_data['title']}")

                safe_filename = sanitize_filename(article_data['title'])
                json_path = json_dir / f"{safe_filename}.json"
                save_to_json(article_data, json_path)

                txt_path = txt_dir / f"{safe_filename}.txt"
                save_to_txt(article_data, txt_path)

                successful_scrapes += 1
                time.sleep(1)

        except FileNotFoundError:
            print(f"Error: File {args.url_file} not found")
            sys.exit(1)

    # Mode 2: Scrape from encyclopedia index
    else:
        # Determine which letters to process
        if args.letters:
            if '-' in args.letters:
                # Range like "A-C"
                start, end = args.letters.split('-')
                letters = [chr(c) for c in range(ord(start.upper()), ord(end.upper()) + 1)]
            else:
                # Individual letters like "ABC"
                letters = list(args.letters.upper())
        else:
            # All letters
            letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + ['0-9']

        print(f"Processing letters: {', '.join(letters)}\n")

        for letter in letters:
            print(f"\n{'=' * 80}")
            print(f"Processing letter: {letter}")
            print('=' * 80)

            # Get article links for this letter
            article_links = get_article_links_from_index(letter)

            if not article_links:
                print(f"  No articles found for {letter}, skipping...")
                continue

            # Limit if in test_images mode
            if args.max_per_letter:
                article_links = article_links[:args.max_per_letter]
                print(f"  Limited to {len(article_links)} articles (--max-per-letter)")

            total_articles += len(article_links)

            # Debug: show first few links
            if args.debug and article_links:
                print(f"\n  Sample URLs:")
                for url in article_links[:3]:
                    print(f"    - {url}")

            # Scrape each article
            for i, url in enumerate(article_links, 1):
                print(f"\n[{letter}] Article {i}/{len(article_links)}: ", end='')

                # Scrape the article
                article_data = scrape_article(url)

                if article_data is None:
                    failed_scrapes += 1
                    continue

                print(f"✓ {article_data['title']}")

                # Generate filename from title
                safe_filename = sanitize_filename(article_data['title'])

                # Save to JSON
                json_path = json_dir / f"{safe_filename}.json"
                save_to_json(article_data, json_path)

                # Save to TXT
                txt_path = txt_dir / f"{safe_filename}.txt"
                save_to_txt(article_data, txt_path)

                successful_scrapes += 1

                # Be nice to the server - add delay
                time.sleep(1)

            print(f"\nCompleted letter {letter}: {len(article_links)} articles processed")

    # Final summary
    print("\n" + "=" * 80)
    print("SCRAPING COMPLETE")
    print("=" * 80)
    print(f"\nTotal articles found: {total_articles}")
    print(f"Successfully scraped: {successful_scrapes}")
    print(f"Failed: {failed_scrapes}")
    print(f"\nFiles saved to:")
    print(f"  JSON: {json_dir.absolute()}")
    print(f"  TXT:  {txt_dir.absolute()}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()