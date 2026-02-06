import requests
from bs4 import BeautifulSoup
import json
from typing import Dict, List, Optional


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


def main():
    """Main function to demonstrate the scraper"""
    scraper = MedlinePlusScraper()

    # Example URL
    url = "https://medlineplus.gov/lab-tests/complete-blood-count-cbc/"

    print(f"Scraping: {url}")
    article_data = scraper.scrape_article(url)

    # Save to JSON
    json_file = "cbc_article.json"
    scraper.save_to_json(article_data, json_file)
    print(f"Saved JSON to: {json_file}")

    # Save to text
    text_file = "cbc_article.txt"
    scraper.save_to_text(article_data, text_file)
    print(f"Saved text to: {text_file}")

    # Print summary
    print(f"\nArticle Title: {article_data['title']}")
    print(f"Number of sections: {len(article_data['sections'])}")
    print(f"Related topics: {len(article_data['related_topics'])}")
    print(f"Related tests: {len(article_data['related_tests'])}")
    print(f"References: {len(article_data['references'])}")


if __name__ == "__main__":
    main()