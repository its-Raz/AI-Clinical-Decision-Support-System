#!/usr/bin/env python3
"""
MedlinePlus Bulk Article Scraper
Crawls the lab tests index page and scrapes all articles
"""

import requests
from bs4 import BeautifulSoup
import json
import os
import time
from typing import Dict, List, Optional
from urllib.parse import urljoin
import re


class MedlinePlusScraper:
    """Scraper for extracting article content from MedlinePlus pages"""

    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    def scrape_article(self, url: str) -> Dict:
        """
        Scrape article content from a MedlinePlus page

        Args:
            url: The URL of the MedlinePlus article

        Returns:
            Dictionary containing structured article data
        """
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        article_data = {
            'url': url,
            'title': self._extract_title(soup),
            'sections': self._extract_sections(soup),
            'related_topics': self._extract_related_topics(soup),
            'related_tests': self._extract_related_tests(soup),
            'references': self._extract_references(soup),
            'last_updated': self._extract_last_updated(soup)
        }

        return article_data

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract the main article title"""
        title_tag = soup.find('h1')
        return title_tag.get_text(strip=True) if title_tag else ""

    def _extract_sections(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract all sections with their headings and content"""
        sections = []

        # Find all h2 headings which represent main sections
        section_headings = soup.find_all('h2')

        for heading in section_headings:
            section_title = heading.get_text(strip=True)
            content_parts = []

            # Get all content until the next h2
            current = heading.find_next_sibling()
            while current and current.name != 'h2':
                if current.name in ['p', 'ul', 'ol']:
                    if current.name == 'p':
                        content_parts.append(current.get_text(strip=True))
                    else:  # ul or ol
                        list_items = [li.get_text(strip=True) for li in current.find_all('li')]
                        content_parts.extend(list_items)
                current = current.find_next_sibling()

            sections.append({
                'title': section_title,
                'content': content_parts
            })

        return sections

    def _extract_related_topics(self, soup: BeautifulSoup) -> List[str]:
        """Extract related health topics"""
        related_topics = []

        # Find the "Related Health Topics" section
        topics_heading = soup.find('h2', string='Related Health Topics')
        if topics_heading:
            topics_list = topics_heading.find_next('ul')
            if topics_list:
                related_topics = [
                    li.get_text(strip=True)
                    for li in topics_list.find_all('li')
                ]

        return related_topics

    def _extract_related_tests(self, soup: BeautifulSoup) -> List[str]:
        """Extract related medical tests"""
        related_tests = []

        # Find the "Related Medical Tests" section
        tests_heading = soup.find('h2', string='Related Medical Tests')
        if tests_heading:
            tests_list = tests_heading.find_next('ul')
            if tests_list:
                related_tests = [
                    li.get_text(strip=True)
                    for li in tests_list.find_all('li')
                ]

        return related_tests

    def _extract_references(self, soup: BeautifulSoup) -> List[str]:
        """Extract references"""
        references = []

        # Find the "References" section
        ref_heading = soup.find('h2', string='References')
        if ref_heading:
            ref_list = ref_heading.find_next('ol')
            if ref_list:
                references = [
                    li.get_text(strip=True)
                    for li in ref_list.find_all('li')
                ]

        return references

    def _extract_last_updated(self, soup: BeautifulSoup) -> str:
        """Extract the last updated date"""
        # Look for text containing "Last updated"
        text_nodes = soup.find_all(string=lambda text: text and 'Last updated' in text)
        if text_nodes:
            return text_nodes[0].strip()
        return ""

    def save_to_json(self, data: Dict, filename: str):
        """Save scraped data to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def save_to_text(self, data: Dict, filename: str):
        """Save scraped data to a formatted text file"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# {data['title']}\n\n")
            f.write(f"URL: {data['url']}\n")
            if data['last_updated']:
                f.write(f"{data['last_updated']}\n\n")
            f.write("---\n\n")

            # Write sections
            for section in data['sections']:
                f.write(f"## {section['title']}\n\n")
                for content in section['content']:
                    f.write(f"{content}\n\n")

            # Write related topics
            if data['related_topics']:
                f.write("## Related Health Topics\n\n")
                for topic in data['related_topics']:
                    f.write(f"- {topic}\n")
                f.write("\n")

            # Write related tests
            if data['related_tests']:
                f.write("## Related Medical Tests\n\n")
                for test in data['related_tests']:
                    f.write(f"- {test}\n")
                f.write("\n")

            # Write references
            if data['references']:
                f.write("## References\n\n")
                for i, ref in enumerate(data['references'], 1):
                    f.write(f"{i}. {ref}\n\n")


class MedlinePlusBulkScraper:
    """Bulk scraper for all lab tests articles"""

    def __init__(self, base_url: str = "https://medlineplus.gov/lab-tests/"):
        self.base_url = base_url
        self.scraper = MedlinePlusScraper()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    def get_all_article_urls(self) -> List[str]:
        """
        Extract all lab test article URLs from the index page

        Returns:
            List of article URLs
        """
        print(f"Fetching index page: {self.base_url}")
        response = requests.get(self.base_url, headers=self.headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        article_urls = []

        # Find all links in the main content area
        # MedlinePlus typically uses <a> tags within the content area
        # Look for links that point to lab-tests articles
        links = soup.find_all('a', href=True)

        for link in links:
            href = link['href']
            # Filter for lab test article URLs
            if '/lab-tests/' in href and href != '/lab-tests/':
                full_url = urljoin(self.base_url, href)
                # Avoid duplicates and filter out non-article pages
                if full_url not in article_urls and not any(x in full_url for x in ['#', '?']):
                    article_urls.append(full_url)

        # Remove the base URL itself if present
        article_urls = [url for url in article_urls if url != self.base_url.rstrip('/')]

        print(f"Found {len(article_urls)} article URLs")
        return article_urls

    def sanitize_filename(self, title: str) -> str:
        """
        Convert article title to a safe filename

        Args:
            title: Article title

        Returns:
            Sanitized filename
        """
        # Remove special characters and replace spaces with underscores
        filename = re.sub(r'[^\w\s-]', '', title)
        filename = re.sub(r'[-\s]+', '_', filename)
        filename = filename.strip('_').lower()
        # Limit length
        return filename[:100]

    def scrape_all_articles(self, output_dir: str = "medlineplus_test_articles", delay: float = 1.0):
        """
        Scrape all articles and save to separate folders

        Args:
            output_dir: Base directory for output
            delay: Delay between requests in seconds (be polite!)
        """
        # Create output directories
        json_dir = os.path.join(output_dir, "json")
        text_dir = os.path.join(output_dir, "text")
        os.makedirs(json_dir, exist_ok=True)
        os.makedirs(text_dir, exist_ok=True)

        # Get all article URLs
        article_urls = self.get_all_article_urls()

        # Track progress
        total = len(article_urls)
        successful = 0
        failed = []

        print(f"\nStarting to scrape {total} articles...")
        print(f"JSON output: {json_dir}")
        print(f"Text output: {text_dir}")
        print("-" * 60)

        for i, url in enumerate(article_urls, 1):
            try:
                print(f"[{i}/{total}] Scraping: {url}")

                # Scrape the article
                article_data = self.scraper.scrape_article(url)

                # Generate filename from title
                if article_data['title']:
                    filename_base = self.sanitize_filename(article_data['title'])
                else:
                    # Fallback to URL-based filename
                    filename_base = url.split('/')[-2] if url.endswith('/') else url.split('/')[-1]

                # Save to JSON
                json_file = os.path.join(json_dir, f"{filename_base}.json")
                self.scraper.save_to_json(article_data, json_file)

                # Save to text
                text_file = os.path.join(text_dir, f"{filename_base}.txt")
                self.scraper.save_to_text(article_data, text_file)

                print(f"  ✓ Saved as: {filename_base}")
                successful += 1

                # Be polite - add delay between requests
                if i < total:
                    time.sleep(delay)

            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
                failed.append({'url': url, 'error': str(e)})

        # Print summary
        print("\n" + "=" * 60)
        print("SCRAPING COMPLETE")
        print("=" * 60)
        print(f"Total articles: {total}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(failed)}")

        if failed:
            print("\nFailed URLs:")
            for item in failed:
                print(f"  - {item['url']}")
                print(f"    Error: {item['error']}")

        # Save failed URLs to a file
        if failed:
            failed_file = os.path.join(output_dir, "failed_urls.json")
            with open(failed_file, 'w', encoding='utf-8') as f:
                json.dump(failed, f, indent=2)
            print(f"\nFailed URLs saved to: {failed_file}")


def main():
    """Main function to run the bulk scraper"""
    # Create bulk scraper
    bulk_scraper = MedlinePlusBulkScraper()

    # Scrape all articles
    # Adjust delay as needed (1.0 second is polite)
    bulk_scraper.scrape_all_articles(
        output_dir="medlineplus_test_articles",
        delay=1.0
    )


if __name__ == "__main__":
    main()