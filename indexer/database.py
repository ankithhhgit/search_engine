import sqlite3
import json
from pathlib import Path

DB_PATH         = Path("data/documents.db")
CLEAN_DATA_DIR  = Path("data/clean")
PAGERANK_PATH   = Path("data/pagerank.json")


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def create_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id          TEXT PRIMARY KEY,
            url             TEXT NOT NULL UNIQUE,
            title           TEXT NOT NULL,
            token_count     INTEGER NOT NULL,
            pagerank_score  REAL NOT NULL DEFAULT 0.0
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_url ON documents(url)")
    conn.commit()


def populate(
    clean_dir: Path = CLEAN_DATA_DIR,
    pagerank_path: Path = PAGERANK_PATH,
    db_path: Path = DB_PATH,
) -> int:
    pagerank_scores: dict[str, float] = {}
    if pagerank_path.exists():
        with open(pagerank_path, encoding="utf-8") as f:
            pagerank_scores = json.load(f)

    files = list(clean_dir.glob("*.json"))
    if not files:
        print("No clean files found.")
        return 0

    conn = get_connection(db_path)
    create_table(conn)

    rows = []
    for filepath in files:
        with open(filepath, encoding="utf-8") as f:
            page = json.load(f)
        rows.append((
            filepath.stem,
            page["url"],
            page["title"],
            page["token_count"],
            pagerank_scores.get(page["url"], 0.0),
        ))

    conn.executemany("""
        INSERT OR REPLACE INTO documents
            (doc_id, url, title, token_count, pagerank_score)
        VALUES (?, ?, ?, ?, ?)
    """, rows)
    conn.commit()
    conn.close()

    print(f"Database populated with {len(rows)} documents → {db_path}")
    return len(rows)


def get_document(doc_id: str, db_path: Path = DB_PATH) -> dict | None:
    conn = get_connection(db_path)
    row = conn.execute(
        "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_pagerank_score(url: str, db_path: Path = DB_PATH) -> float:
    conn = get_connection(db_path)
    row = conn.execute(
        "SELECT pagerank_score FROM documents WHERE url = ?", (url,)
    ).fetchone()
    conn.close()
    return row["pagerank_score"] if row else 0.0


def get_stats(db_path: Path = DB_PATH) -> dict:
    conn = get_connection(db_path)
    row = conn.execute("""
        SELECT
            COUNT(*)        AS total_docs,
            AVG(token_count)        AS avg_token_count,
            AVG(pagerank_score)     AS avg_pagerank,
            MAX(pagerank_score)     AS max_pagerank
        FROM documents
    """).fetchone()
    conn.close()
    return dict(row)


if __name__ == "__main__":
    print("Building database...\n")
    count = populate()

    print("\n=== Database stats ===")
    stats = get_stats()
    for k, v in stats.items():
        print(f"  {k:<20} : {round(v, 4) if isinstance(v, float) else v}")

    print("\n=== Top 5 by PageRank ===")
    conn = get_connection()
    rows = conn.execute("""
        SELECT url, title, pagerank_score
        FROM documents
        ORDER BY pagerank_score DESC
        LIMIT 5
    """).fetchall()
    conn.close()
    for row in rows:
        short = row["url"].replace("https://books.toscrape.com", "")
        print(f"  [{row['pagerank_score']:.4f}]  {row['title'][:45]}")