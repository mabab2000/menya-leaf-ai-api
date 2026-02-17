import os
from sqlalchemy import create_engine, text

DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres.jwdhnfdgkokpyvrgmtxd:c01rJFqtYpcy8JU7@aws-1-eu-west-3.pooler.supabase.com:6543/postgres")

engine = create_engine(DB_URL)

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS result (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    image_path TEXT NOT NULL,
    result JSONB,
    created_at TIMESTAMP DEFAULT now()
)
"""


def init_db():
    with engine.begin() as conn:
        conn.execute(text(CREATE_SQL))


if __name__ == "__main__":
    try:
        init_db()
        print("Migration completed: 'result' table created/ensured.")
    except Exception as e:
        print("Migration failed:", e)
        raise
