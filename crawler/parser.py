import hashlib
import json
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

RAW_DATA_DIR = Path("data/raw")


def parse(html: str, url: str) -> dict:
    soup = BeautifulSoup(html, "lxml")

    title = _extract_title(soup)
    raw_text = _extract_text(soup)
    links = extract_links(soup, url)

    return {
        "url": url,
        "title": title,
        "raw_text": raw_text,
        "links": links,
    }


def _extract_title(soup: BeautifulSoup) -> str:
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)
    return "Untitled"


def _extract_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    body = soup.find("body")
    if not body:
        return ""

    raw = body.get_text(separator=" ")
    return " ".join(raw.split())


def extract_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    links = []
    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)

        if parsed.scheme not in ("http", "https"):
            continue
        if not parsed.netloc:
            continue

        clean = parsed._replace(fragment="").geturl()
        links.append(clean)

    return list(set(links))


def save_raw(page: dict) -> Path:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    url_hash = hashlib.md5(page["url"].encode()).hexdigest()
    filepath = RAW_DATA_DIR / f"{url_hash}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(page, f, indent=2, ensure_ascii=False)
    return filepath


if __name__ == "__main__":
    from crawler.fetcher import fetch

    url = "https://books.toscrape.com"
    html = fetch(url)
    if html:
        page = parse(html, url)
        path = save_raw(page)
        print(f"Title   : {page['title']}")
        print(f"Text    : {page['raw_text'][:120]}...")
        print(f"Links   : {len(page['links'])} found")
        print(f"Saved   : {path}")