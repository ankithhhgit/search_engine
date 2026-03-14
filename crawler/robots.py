import time
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

BOT_NAME = "SearchEngineBot"
DEFAULT_CRAWL_DELAY = 1.0


class RobotsCache:
    def __init__(self):
        self._parsers: dict[str, RobotFileParser] = {}
        self._last_request_time: dict[str, float] = {}

    def can_fetch(self, url: str) -> bool:
        domain = self._get_domain(url)
        parser = self._get_parser(domain, url)
        if parser is None:
            return True
        return parser.can_fetch(BOT_NAME, url)

    def wait_if_needed(self, url: str) -> None:
        domain = self._get_domain(url)
        last = self._last_request_time.get(domain, 0)
        delay = self._get_crawl_delay(domain)
        elapsed = time.time() - last
        remaining = delay - elapsed

        if remaining > 0:
            print(f"     Waiting {remaining:.2f}s for {domain}")
            time.sleep(remaining)

        self._last_request_time[domain] = time.time()

    def _get_crawl_delay(self, domain: str) -> float:
        parser = self._parsers.get(domain)
        if parser is None:
            return DEFAULT_CRAWL_DELAY
        delay = parser.crawl_delay(BOT_NAME)
        return float(delay) if delay is not None else DEFAULT_CRAWL_DELAY

    def _get_parser(self, domain: str, url: str) -> RobotFileParser | None:
        if domain in self._parsers:
            return self._parsers[domain]

        robots_url = self._build_robots_url(url)
        parser = RobotFileParser()
        parser.set_url(robots_url)

        try:
            parser.read()
            print(f"     Loaded robots.txt for {domain}")
        except Exception:
            print(f"     No robots.txt found for {domain}, allowing all")
            self._parsers[domain] = None
            return None

        self._parsers[domain] = parser
        return parser

    def _get_domain(self, url: str) -> str:
        return urlparse(url).netloc

    def _build_robots_url(self, url: str) -> str:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}/robots.txt"


if __name__ == "__main__":
    robots = RobotsCache()

    test_urls = [
        "https://books.toscrape.com",
        "https://books.toscrape.com/catalogue/page-2.html",
        "https://books.toscrape.com/admin",
    ]

    print("Testing robots.txt rules:\n")
    for url in test_urls:
        allowed = robots.can_fetch(url)
        status = "ALLOWED" if allowed else "BLOCKED"
        print(f"  [{status}] {url}")

    print("\nTesting rate limiting (crawling 3 pages):\n")
    from crawler.fetcher import fetch
    from crawler.parser import parse

    for url in test_urls[:3]:
        if robots.can_fetch(url):
            robots.wait_if_needed(url)
            html = fetch(url)
            if html:
                page = parse(html, url)
                print(f"  Fetched: {page['title']}")