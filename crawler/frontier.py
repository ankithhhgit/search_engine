from collections import deque
from urllib.parse import urlparse


class URLFrontier:
    def __init__(self, seed_url: str):
        self.seed_domain = urlparse(seed_url).netloc
        self._queue: deque[str] = deque()
        self._visited: set[str] = set()
        self._queued: set[str] = set()

        self.add(seed_url)

    def add(self, url: str) -> bool:
        if not self._is_same_domain(url):
            return False
        if self.is_visited(url):
            return False
        if url in self._queued:
            return False

        self._queue.append(url)
        self._queued.add(url)
        return True

    def pop(self) -> str | None:
        if not self._queue:
            return None
        url = self._queue.popleft()
        self._visited.add(url)
        return url

    def add_many(self, urls: list[str]) -> int:
        return sum(1 for url in urls if self.add(url))

    def is_visited(self, url: str) -> bool:
        return url in self._visited

    def is_empty(self) -> bool:
        return len(self._queue) == 0

    def stats(self) -> dict:
        return {
            "queued": len(self._queue),
            "visited": len(self._visited),
            "total_seen": len(self._visited) + len(self._queue),
        }

    def _is_same_domain(self, url: str) -> bool:
        return urlparse(url).netloc == self.seed_domain


if __name__ == "__main__":
    from crawler.fetcher import fetch
    from crawler.parser import parse, extract_links

    seed = "https://books.toscrape.com"
    frontier = URLFrontier(seed)

    print(f"Starting crawl from: {seed}")
    print(f"Domain locked to: {frontier.seed_domain}\n")

    pages_crawled = 0
    max_pages = 5

    while not frontier.is_empty() and pages_crawled < max_pages:
        url = frontier.pop()
        print(f"[{pages_crawled + 1}] Fetching: {url}")

        html = fetch(url)
        if not html:
            print(f"     Skipped — fetch failed\n")
            continue

        page = parse(html, url)
        added = frontier.add_many(page["links"])

        pages_crawled += 1
        print(f"     Title : {page['title']}")
        print(f"     Links : {len(page['links'])} found, {added} new added to queue")
        print(f"     Stats : {frontier.stats()}\n")

    print(f"Done. Crawled {pages_crawled} pages.")
    print(f"Final stats: {frontier.stats()}")