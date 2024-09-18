import sqlite3
import pickle
import io
import torch
from model import SimpleModel

class DBHandler:
    def __init__(self, db_path='models.db'):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY,
                    model_data BLOB,
                    optimizer_data BLOB
                )
            ''')
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    level TEXT,
                    message TEXT
                )
            ''')

    def save_model(self, model_id, model, optimizer):
        model_buffer = io.BytesIO()
        torch.save(model.state_dict(), model_buffer)
        optimizer_buffer = io.BytesIO()
        torch.save(optimizer.state_dict(), optimizer_buffer)
        
        with self.conn:
            self.conn.execute('''
                INSERT OR REPLACE INTO models (id, model_data, optimizer_data)
                VALUES (?, ?, ?)
            ''', (model_id, model_buffer.getvalue(), optimizer_buffer.getvalue()))

    def load_model(self, model_id):
        with self.conn:
            cursor = self.conn.execute('SELECT model_data, optimizer_data FROM models WHERE id = ?', (model_id,))
            row = cursor.fetchone()
            if row:
                model_data, optimizer_data = row
                model = SimpleModel()
                model.load_state_dict(torch.load(io.BytesIO(model_data)))
                optimizer = torch.optim.Adam(model.parameters())
                optimizer.load_state_dict(torch.load(io.BytesIO(optimizer_data)))
                return model, optimizer
            return None, None

    def save_log(self, timestamp, level, message):
        with self.conn:
            self.conn.execute('''
                INSERT INTO logs (timestamp, level, message)
                VALUES (?, ?, ?)
            ''', (timestamp, level, message))

    def get_logs(self, limit=100):
        with self.conn:
            cursor = self.conn.execute('SELECT * FROM logs ORDER BY id DESC LIMIT ?', (limit,))
            return cursor.fetchall()

    def close(self):
        self.conn.close()