import json
import time
from pathlib import Path

from crawler.fetcher import fetch
from crawler.frontier import URLFrontier
from crawler.parser import parse, save_raw
from indexer.text_processor import process_page, _save_clean

CLEAN_DATA_DIR = Path("data/clean")


def crawl(seed_url: str, max_pages: int = 100) -> list[dict]:
    frontier = URLFrontier(seed_url)
    crawled_pages = []
    failed_urls = []

    print(f"Starting crawl from : {seed_url}")
    print(f"Max pages           : {max_pages}")
    print(f"Domain locked to    : {frontier.seed_domain}")
    print("-" * 50)

    while not frontier.is_empty() and len(crawled_pages) < max_pages:
        url = frontier.pop()
        current = len(crawled_pages) + 1
        print(f"[{current}/{max_pages}] {url}")

        html = fetch(url)
        if not html:
            failed_urls.append(url)
            print(f"  Failed — skipping\n")
            continue

        page = parse(html, url)
        raw_path = save_raw(page)

        clean_page = process_page(page)
        _save_clean(clean_page, raw_path.stem)

        added = frontier.add_many(page["links"])
        crawled_pages.append(clean_page)

        print(f"  Title      : {clean_page['title']}")
        print(f"  Tokens     : {clean_page['token_count']}")
        print(f"  New links  : {added} added to queue")
        print(f"  Queue size : {frontier.stats()['queued']}")
        print()

    _print_summary(crawled_pages, failed_urls, frontier)
    return crawled_pages


def _print_summary(pages: list[dict], failed: list[str], frontier: URLFrontier) -> None:
    total_tokens = sum(p["token_count"] for p in pages)
    avg_tokens = total_tokens // len(pages) if pages else 0

    print("=" * 50)
    print("CRAWL COMPLETE")
    print("=" * 50)
    print(f"Pages crawled     : {len(pages)}")
    print(f"Pages failed      : {len(failed)}")
    print(f"Total tokens      : {total_tokens:,}")
    print(f"Avg tokens/page   : {avg_tokens:,}")
    print(f"URLs remaining    : {frontier.stats()['queued']}")
    print()

    if failed:
        print(f"Failed URLs:")
        for url in failed:
            print(f"  {url}")
        print()

    print(f"Data saved to     : data/raw/ and data/clean/")


if __name__ == "__main__":
    crawl(
        seed_url="https://books.toscrape.com",
        max_pages=100,
    )