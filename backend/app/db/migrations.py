from sqlalchemy import text


def _get_columns(conn, table_name):
    rows = conn.execute(text("PRAGMA table_info(%s)" % table_name)).fetchall()
    return set([row[1] for row in rows])


def _add_column_if_missing(conn, table_name, column_name, ddl):
    cols = _get_columns(conn, table_name)
    if column_name not in cols:
        conn.execute(text("ALTER TABLE %s ADD COLUMN %s" % (table_name, ddl)))


def run_migrations(engine):
    with engine.begin() as conn:
        _add_column_if_missing(conn, "documents", "process_status", "process_status VARCHAR(30) NOT NULL DEFAULT 'uploaded'")
        _add_column_if_missing(conn, "documents", "process_error", "process_error TEXT NOT NULL DEFAULT ''")
        _add_column_if_missing(conn, "documents", "chunk_count", "chunk_count INTEGER NOT NULL DEFAULT 0")
        _add_column_if_missing(conn, "documents", "processed_at", "processed_at DATETIME")
        _add_column_if_missing(conn, "documents", "file_hash", "file_hash VARCHAR(64) NOT NULL DEFAULT ''")
        _add_column_if_missing(conn, "documents", "markdown_path", "markdown_path VARCHAR(1024) NOT NULL DEFAULT ''")
        _add_column_if_missing(conn, "documents", "markdown_url", "markdown_url VARCHAR(1024) NOT NULL DEFAULT ''")
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_documents_file_hash ON documents (file_hash)"))

        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY,
                    document_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    embedding_json TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(id)
                )
                """
            )
        )
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_chunks_document_id ON chunks (document_id)"))
