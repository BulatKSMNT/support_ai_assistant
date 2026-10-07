import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path


class Repository:
    """Canonical document versions. Uncommitted vector points never become searchable."""

    def __init__(self, path: Path, signature: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.connection() as db, db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, source_key TEXT UNIQUE NOT NULL, title TEXT NOT NULL,
                    source_url TEXT, version INTEGER NOT NULL, content_hash TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id)
                    ON DELETE CASCADE, ordinal INTEGER NOT NULL, text TEXT NOT NULL, page INTEGER
                );
                CREATE INDEX IF NOT EXISTS chunks_document ON chunks(document_id);
            """)
            previous = db.execute("SELECT value FROM metadata WHERE key='signature'").fetchone()
            if previous and previous[0] != signature:
                raise RuntimeError(
                    "Index configuration changed. Use a new RAG_DATA_DIR and re-import documents."
                )
            db.execute("INSERT OR IGNORE INTO metadata VALUES ('signature', ?)", (signature,))

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
        finally:
            db.close()

    def documents(self) -> list[dict]:
        with self.connection() as db:
            return [dict(row) for row in db.execute("SELECT * FROM documents ORDER BY title")]

    def document(self, source_key: str) -> dict | None:
        with self.connection() as db:
            row = db.execute("SELECT * FROM documents WHERE source_key=?", (source_key,)).fetchone()
            return dict(row) if row else None

    def chunks(self) -> list[dict]:
        with self.connection() as db:
            return [
                dict(row)
                for row in db.execute("""
                SELECT c.id, c.document_id, c.text, c.page, c.ordinal,
                       d.title, d.source_key, d.source_url, d.version
                FROM chunks c JOIN documents d ON c.document_id=d.id
                ORDER BY d.source_key, c.ordinal
            """)
            ]

    def replace(self, document: dict, chunks: list[dict]) -> None:
        with self.connection() as db, db:
            db.execute(
                """
                INSERT INTO documents VALUES
                    (:id,:source_key,:title,:source_url,:version,:content_hash,:chunk_count,
                     :created_at,:updated_at)
                ON CONFLICT(id) DO UPDATE SET title=excluded.title, source_url=excluded.source_url,
                    version=excluded.version, content_hash=excluded.content_hash,
                    chunk_count=excluded.chunk_count, updated_at=excluded.updated_at
            """,
                document,
            )
            db.execute("DELETE FROM chunks WHERE document_id=?", (document["id"],))
            db.executemany(
                """
                INSERT INTO chunks(id,document_id,ordinal,text,page)
                VALUES (:id,:document_id,:ordinal,:text,:page)
            """,
                chunks,
            )

    def delete(self, document_id: str) -> bool:
        with self.connection() as db, db:
            return db.execute("DELETE FROM documents WHERE id=?", (document_id,)).rowcount > 0

    def export(self) -> str:
        return json.dumps(
            {"documents": self.documents(), "chunks": self.chunks()}, ensure_ascii=False
        )
