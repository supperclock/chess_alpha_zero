import sqlite3
import pickle

class TTCacher:
    def __init__(self, db_path="tt_cache.sqlite"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_table()

    def _init_table(self):
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS transposition (
            zobrist TEXT PRIMARY KEY,
            value REAL,
            depth INTEGER,
            flag TEXT,
            best_move BLOB
        )
        """)
        self.conn.commit()

    def get(self, zob):
        cur = self.conn.cursor()
        zob_str = str(zob)
        cur.execute("SELECT value, depth, flag, best_move FROM transposition WHERE zobrist=?", (zob_str,))
        row = cur.fetchone()
        if row:
            val, depth, flag, best_move_blob = row
            best_move = pickle.loads(best_move_blob) if best_move_blob else None
            return {"value": val, "depth": depth, "flag": flag, "best_move": best_move}
        return None

    def put(self, zob, value, depth, flag, best_move):
        best_move_blob = pickle.dumps(best_move) if best_move else None
        cur = self.conn.cursor()
        zob_str = str(zob)
        cur.execute("""
        INSERT INTO transposition (zobrist, value, depth, flag, best_move)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(zobrist) DO UPDATE SET
            value=excluded.value,
            depth=excluded.depth,
            flag=excluded.flag,
            best_move=excluded.best_move
        """, (zob_str, value, depth, flag, best_move_blob))
        self.conn.commit()

    def close(self):
        self.conn.close()
