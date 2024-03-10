import sqlite3
from pathlib import Path
from typing import Dict, Set, Optional

from utils.parquet_utils import parquet_file_iterator
from tokenizer import tokenize


def _extract_tokens(record: Dict) -> Set[str]:
    text = record['text']
    tokens = tokenize(text)
    return set(tokens)


class TokenIndexer:

    def __init__(self):
        self.conn = sqlite3.connect('../db/token_index.db')
        self.cursor = self.conn.cursor()
        self.cursor.execute(
            """CREATE TABLE IF NOT EXISTS token_index (
                    idx INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                    token TEXT NOT NULL,
                    doc_amount INTEGER NOT NULL
                );""")
        self.cursor.execute("""CREATE UNIQUE INDEX IF NOT EXISTS token_idx ON token_index (token);""")

        print("Loading token index...")
        self.token_2_idx = dict()
        self.idx_2_amount = dict()

        for idx, token, doc_amount in self.cursor.execute(
                """SELECT idx, token, doc_amount FROM token_index;""").fetchall():
            self.token_2_idx[token] = idx
            self.idx_2_amount[idx] = doc_amount
        print("Token index loaded")
        self.token_count = dict()
        self.n_docs = sum(v for v in self.idx_2_amount.values())

    def import_data(self, path: Path):
        msg = f"Reading file '{path.name}'..."
        for record in parquet_file_iterator(str(path.absolute()), ["text"], 1000, msg):
            tokens = _extract_tokens(record)
            for token in tokens:
                if token not in self.token_count:
                    self.token_count[token] = 0
                self.token_count[token] += 1

    def index_of(self, token: str) -> Optional[int]:
        # result = self.cursor.execute("""SELECT idx FROM token_index WHERE token = ?;""", [token]).fetchone()
        # if result is not None:
        #     return result[0]
        # else:
        #     return None
        return self.token_2_idx.get(token, None)

    def doc_amount_of(self, token_idx: int) -> Optional[int]:
        return self.idx_2_amount[token_idx]

    def size(self):
        # result = self.cursor.execute("""SELECT COUNT(*) FROM token_index;""").fetchone()
        # if result is not None:
        #     return result[0]
        # else:
        #     return None
        return len(self.token_2_idx)

    def commit(self, limit: int):
        self.cursor.executemany(
            """INSERT INTO token_index (token, doc_amount) VALUES (?, ?);""",
            [[token, doc_amount] for token, doc_amount in self.token_count.items() if doc_amount > limit]
        )
        self.conn.commit()

    def close(self):
        self.conn.close()


if __name__ == '__main__':
    file = Path('../wikidump')

    ig = TokenIndexer()
    for f in file.iterdir():
        ig.import_data(f)
    ig.commit(50)
    print("loaded")
