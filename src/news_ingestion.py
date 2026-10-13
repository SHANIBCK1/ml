"""
Modular news ingestion module for NewsInsight AI.
Provides safe, legitimate RSS feed fetching from public news syndications (e.g., BBC, Reuters feeds)
with robust timeout handling, offline fallbacks, and distinction between fetched, stored, and analyzed articles.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
import xml.etree.ElementTree as ET
import urllib.request
import urllib.error

# Public RSS feeds with open syndication
DEFAULT_RSS_FEEDS = {
    "BBC World": "http://feeds.bbci.co.uk/news/world/rss.xml",
    "BBC Tech": "http://feeds.bbci.co.uk/news/technology/rss.xml",
    "BBC Business": "http://feeds.bbci.co.uk/news/business/rss.xml",
}


def fetch_rss_feed(feed_url: str, source_name: str = "RSS", max_items: int = 10, timeout: int = 5) -> List[Dict[str, Any]]:
    """
    Fetches and parses a public RSS feed via standard HTTP request.
    Returns standardized raw news items:
    [{"headline": str, "article_text": str, "source": str, "published_at": str}]
    """
    items = []
    try:
        req = urllib.request.Request(
            feed_url,
            headers={"User-Agent": "NewsInsight-AI/1.0 (Educational Academic Project)"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            xml_data = response.read()

        root = ET.fromstring(xml_data)
        # Standard RSS 2.0 structure: <rss><channel><item>
        channel = root.find("channel")
        if channel is None:
            return items

        for item_node in channel.findall("item")[:max_items]:
            title_node = item_node.find("title")
            desc_node = item_node.find("description")
            pub_date_node = item_node.find("pubDate")

            headline = title_node.text.strip() if title_node is not None and title_node.text else ""
            desc = desc_node.text.strip() if desc_node is not None and desc_node.text else headline
            pub_date = pub_date_node.text.strip() if pub_date_node is not None and pub_date_node.text else datetime.utcnow().strftime("%Y-%m-%d")

            if headline:
                items.append({
                    "headline": headline,
                    "article_text": desc,
                    "source": source_name,
                    "published_at": pub_date
                })

    except (urllib.error.URLError, TimeoutError, ET.ParseError, Exception) as e:
        # Gracefully handle network unavailability or parsing error
        pass

    return items


def fetch_sample_live_batch(max_per_feed: int = 5) -> List[Dict[str, Any]]:
    """
    Fetches articles across default RSS feeds.
    If internet is offline or feeds are unreachable, returns empty list without error.
    """
    all_articles = []
    for source_name, url in DEFAULT_RSS_FEEDS.items():
        articles = fetch_rss_feed(url, source_name=source_name, max_items=max_per_feed)
        all_articles.extend(articles)
    return all_articles
