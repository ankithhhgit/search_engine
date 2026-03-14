import httpx

HEADERS = {
    "User-Agent": "SearchEngineBot/1.0 (learning project)"
}


def fetch(url: str) -> str | None:
    try:
        response = httpx.get(url, headers=HEADERS, timeout=10, follow_redirects=True)
        response.raise_for_status()
        return response.text
    except httpx.HTTPStatusError as e:
        print(f"HTTP error {e.response.status_code} for {url}")
        return None
    except httpx.RequestError as e:
        print(f"Request failed for {url}: {e}")
        return None