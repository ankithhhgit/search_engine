import json
import re
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

RAW_DATA_DIR = Path("data/raw")
CLEAN_DATA_DIR = Path("data/clean")

_stemmer = PorterStemmer()
_stop_words = set(stopwords.words("english"))


def process(text: str) -> list[str]:
    text = _lowercase(text)
    text = _remove_noise(text)
    tokens = _tokenize(text)
    tokens = _remove_stopwords(tokens)
    tokens = _keep_alpha(tokens)
    tokens = _stem(tokens)
    return tokens


def process_page(raw_page: dict) -> dict:
    tokens = process(raw_page["raw_text"])
    return {
        "url": raw_page["url"],
        "title": raw_page["title"],
        "links": raw_page.get("links", []),
        "tokens": tokens,
        "token_count": len(tokens),
    }


def process_all() -> list[dict]:
    CLEAN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_files = list(RAW_DATA_DIR.glob("*.json"))

    if not raw_files:
        print("No raw files found. Run the crawler first.")
        return []

    results = []
    for filepath in raw_files:
        with open(filepath, encoding="utf-8") as f:
            raw_page = json.load(f)

        clean_page = process_page(raw_page)
        _save_clean(clean_page, filepath.stem)
        results.append(clean_page)

    return results


def _save_clean(page: dict, filename_stem: str) -> Path:
    CLEAN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    filepath = CLEAN_DATA_DIR / f"{filename_stem}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(page, f, indent=2, ensure_ascii=False)
    return filepath


def _lowercase(text: str) -> str:
    return text.lower()


def _remove_noise(text: str) -> str:
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokenize(text: str) -> list[str]:
    return word_tokenize(text)


def _remove_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in _stop_words]


def _keep_alpha(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t.isalpha()]


def _stem(tokens: list[str]) -> list[str]:
    return [_stemmer.stem(t) for t in tokens]


if __name__ == "__main__":
    sample_text = """
    Learning Python is really exciting! Python provides powerful libraries
    for web crawling, searching, and indexing documents. The quick brown fox
    jumped over the lazy dogs. Visit https://example.com for more details!!!
    Running, runs, runner all stem to the same root word.
    """

    print("=== Pipeline demo ===\n")
    print(f"Input text:\n{sample_text.strip()}\n")

    text = _lowercase(sample_text)
    print(f"After lowercase:\n{text.strip()}\n")

    text = _remove_noise(text)
    print(f"After noise removal:\n{text.strip()}\n")

    tokens = _tokenize(text)
    print(f"After tokenise ({len(tokens)} tokens):\n{tokens}\n")

    tokens = _remove_stopwords(tokens)
    print(f"After stop word removal ({len(tokens)} tokens):\n{tokens}\n")

    tokens = _keep_alpha(tokens)
    print(f"After keep alpha ({len(tokens)} tokens):\n{tokens}\n")

    tokens = _stem(tokens)
    print(f"After stemming ({len(tokens)} tokens):\n{tokens}\n")

    print("\n=== Processing saved raw files ===\n")
    pages = process_all()
    if pages:
        print(f"\nProcessed {len(pages)} pages\n")
        sample = pages[0]
        print(f"URL         : {sample['url']}")
        print(f"Title       : {sample['title']}")
        print(f"Token count : {sample['token_count']}")
        print(f"First 20    : {sample['tokens'][:20]}")