#!/usr/bin/env python3
"""
Parse MedlinePlus health topics XML and convert to JSON format.
Extracts English language topics with specified fields.
"""

import xml.etree.ElementTree as ET
import json
from pathlib import Path


def parse_health_topics(xml_file_path):
    """
    Parse health topics XML file and extract English topics.

    Args:
        xml_file_path: Path to the XML file

    Returns:
        List of dictionaries containing health topic information
    """
    # Parse the XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    health_topics = []

    # Iterate through all health-topic elements
    for topic in root.findall('health-topic'):
        # Only process English language topics
        if topic.get('language') == 'English':
            topic_data = {
                'title': topic.get('title', ''),
                'meta-desc': topic.get('meta-desc', ''),
                'also-called': [],
                'full-summary': '',
                'group': [],
                'related-topic': []
            }

            # Extract all "also-called" elements
            for also_called in topic.findall('also-called'):
                if also_called.text:
                    topic_data['also-called'].append(also_called.text)

            # Extract full-summary
            full_summary = topic.find('full-summary')
            if full_summary is not None and full_summary.text:
                topic_data['full-summary'] = full_summary.text

            # Extract all "group" elements
            for group in topic.findall('group'):
                group_data = {
                    'title': group.get('url', '').split('/')[-1].replace('.html', ''),
                    'url': group.get('url', ''),
                    'id': group.get('id', '')
                }
                topic_data['group'].append(group_data)

            # Extract all "related-topic" elements
            for related in topic.findall('related-topic'):
                related_data = {
                    'title': related.get('url', '').split('/')[-1].replace('.html', ''),
                    'url': related.get('url', ''),
                    'id': related.get('id', '')
                }
                topic_data['related-topic'].append(related_data)

            health_topics.append(topic_data)

    return health_topics


def main():
    """Main function to parse XML and save to JSON."""
    # Input and output paths
    xml_file = Path(r'C:\Users\razbi\PycharmProjects\AutonomousClinicalSystem\data_pipeline\data\medline_health_topics.xml')
    output_file = Path(r'C:\Users\razbi\PycharmProjects\AutonomousClinicalSystem\data_pipeline\data\medline_health_topics.json')

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Parsing XML file: {xml_file}")

    # Parse the XML file
    health_topics = parse_health_topics(xml_file)

    print(f"Found {len(health_topics)} English health topics")

    # Save to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(health_topics, f, indent=2, ensure_ascii=False)

    print(f"Successfully saved to: {output_file}")

    # Print sample of first topic
    if health_topics:
        print("\nSample of first topic:")
        print(json.dumps(health_topics[0], indent=2, ensure_ascii=False)[:500] + "...")


if __name__ == '__main__':
    main()