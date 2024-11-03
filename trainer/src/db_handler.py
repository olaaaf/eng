import io
import sqlite3

import torch

from train.model import SimpleModel


class DBHandler:
    def __init__(self, db_path="models.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY,
                    model_data BLOB,
                    optimizer_data BLOB
                )
            """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id INTEGER,
                    timestamp TEXT,
                    level TEXT,
                    message TEXT
                    FOREIGN KEY (model_id) REFERENCES models (id)
                )
            """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS recordings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id INTEGER,
                    timestamp TEXT,
                    path TEXT,
                    FOREIGN KEY (model_id) REFERENCES models (id)
                )
            """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id INTEGER,
                    timestamp TEXT,
                    path TEXT,
                    x_positions BLOB,
                    y_positions BLOB,
                    time INTEGER,
                    died BOOLEAN,
                    FOREIGN KEY (model_id) REFERENCES models (id)
                )
            """
            )


    def save_model(self, model_id, model, optimizer):
        model_buffer = io.BytesIO()
        torch.save(model.state_dict(), model_buffer)
        optimizer_buffer = io.BytesIO()
        torch.save(optimizer.state_dict(), optimizer_buffer)

        with self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO models (id, model_data, optimizer_data)
                VALUES (?, ?, ?)
            """,
                (model_id, model_buffer.getvalue(), optimizer_buffer.getvalue()),
            )

    def load_model(self, model_id):
        with self.conn:
            cursor = self.conn.execute(
                "SELECT model_data, optimizer_data FROM models WHERE id = ?",
                (model_id,),
            )
            row = cursor.fetchone()
            if row:
                model_data, optimizer_data = row
                model = SimpleModel()
                model.load_state_dict(torch.load(io.BytesIO(model_data)))
                optimizer = torch.optim.Adam(model.parameters())
                optimizer.load_state_dict(torch.load(io.BytesIO(optimizer_data)))
                return model, optimizer
            return None, None

    def save_log(self, timestamp, level, message, model_id):
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO logs (timestamp, level, message)
                VALUES (?, ?, ?)
            """,
                (timestamp, level, message),
            )

    def get_logs(self, limit=100):
        with self.conn:
            cursor = self.conn.execute(
                "SELECT * FROM logs ORDER BY id DESC LIMIT ?", (limit,)
            )
            return cursor.fetchall()

    def get_model_logs(self, model_id):
        with self.conn:
            cursor = self.conn.execute(
                "SELECT * FROM logs WHERE model_id is ? ORDER BY id DESC", (model_id)
            )
            return cursor.fetchall()

    def get_recordings_list(self, model_id):
        with self.conn:
            cursor = self.conn.execute(
                "SELECT (id, timestamp, path)  FROM recrodings WHERE model_id is ? ORDER BY id DESC", (model_id)
            )
            return cursor.fetchall()

    def get_recording(self, id):
        pass

    def get_models(self):
        with self.conn:
            cursor = self.conn.execute(
                "SELECT id FROM models"
            )
            return cursor.fetchall()

    def close(self):
        self.conn.close()

    def save_recording(self, model_id):
        pass

    def save_results(self):
        pass

    def get_results_list(self, model_id):
        pass

    def get_results(self, results_id):
        pass
