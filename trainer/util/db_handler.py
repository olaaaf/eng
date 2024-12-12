import io
import json
import logging
import sqlite3
import math

import torch

from train.helpers import Reward
from train.model import SimpleModel


class DBHandler:
    def __init__(self, db_path="models.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()
        self.logger: logging.Logger

    def init_logger(self, logger: logging.Logger):
        self.logger = logger

    def create_tables(self):
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY,
                    train_count INTEGER,
                    model_data BLOB,
                    optimizer_data BLOB,
                    epsilon FLOAT,
                    episode INTEGER
                )
            """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS model_archives (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id INTEGER,
                    model_data BLOB,
                    optimizer_data BLOB,
                    FOREIGN KEY (model_id) REFERENCES models (id)
                )
            """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    level TEXT,
                    message TEXT
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
                    x_positions BLOB,
                    y_positions BLOB,
                    time INTEGER,
                    died BOOLEAN,
                    FOREIGN KEY (model_id) REFERENCES models (id)
                )
            """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS highscores(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id INTEGER,
                    timestamp TEXT,
                    x INTEGER,
                    score INTEGER,
                    time INTEGER,
                    FOREIGN KEY (model_id) REFERENCES models (id)
                )
            """
            )

    def increase_train_count(self, model_id):
        with self.conn:
            self.conn.execute(
                """
                UPDATE models
                SET train_count = COALESCE(train_count, 0) + 1
                WHERE id = ?
                """,
                (model_id,),
            )

    def save_model(self, epsilon, model_id, model, optimizer, episode):
        model_buffer = io.BytesIO()
        if model:
            torch.save(model.state_dict(), model_buffer)
        optimizer_buffer = io.BytesIO()
        if optimizer:
            torch.save(optimizer.state_dict(), optimizer_buffer)

        with self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO models (id, train_count, model_data, optimizer_data, epsilon, episode)
                VALUES (?, 0, ?, ?, ?, ?)
            """,
                (
                    model_id,
                    model_buffer.getvalue(),
                    optimizer_buffer.getvalue(),
                    epsilon,
                    episode,
                ),
            )

    def save_model_archive(self, model_id, model, optimizer):
        model_buffer = io.BytesIO()
        if model:
            torch.save(model.state_dict(), model_buffer)
        optimizer_buffer = io.BytesIO()
        if optimizer:
            torch.save(optimizer.state_dict(), optimizer_buffer)

        with self.conn:
            self.conn.execute(
                """
                INSERT INTO model_archives (model_id, model_data, optimizer_data)
                VALUES (?, ?, ?)
            """,
                (
                    model_id,
                    model_buffer.getvalue(),
                    optimizer_buffer.getvalue(),
                ),
            )

    def get_model_archives(self, model_id):
        with self.conn:
            cursor = self.conn.execute("SELECT id FROM model_archives")
            return cursor.fetchall()

    def load_model(
        self, model_id, reward_handler: Reward
    ) -> tuple[int, SimpleModel | None, torch.optim.Optimizer | None, float, int]:
        with self.conn:
            cursor = self.conn.execute(
                "SELECT train_count, model_data, optimizer_data, epsilon, episode FROM models WHERE id = ?",
                (model_id,),
            )
            row = cursor.fetchone()
            if row:
                times_trained, model_data, optimizer_data, epsilon, episode = row
                model: SimpleModel
                model = SimpleModel(reward_handler)
                model.load_state_dict(
                    torch.load(io.BytesIO(model_data), weights_only=True)
                )
                for param in model.parameters():
                    param.requires_grad = False
                optimizer = torch.optim.Adam(model.parameters())
                optimizer.load_state_dict(
                    torch.load(io.BytesIO(optimizer_data), weights_only=True)
                )
                return times_trained, model, optimizer, epsilon, episode
            return 0, None, None, 1, 0

    def load_model_arhive(
        self, id
    ) -> tuple[SimpleModel | None, torch.optim.Optimizer | None]:
        with self.conn:
            cursor = self.conn.execute(
                "SELECT model_data, optimizer_data FROM model_archives WHERE id = ?",
                (id,),
            )
            row = cursor.fetchone()
            if row:
                model_data, _ = row
                model = SimpleModel()
                model.load_state_dict(
                    torch.load(io.BytesIO(model_data), weights_only=True)
                )
                return model, None
            return None, None

    def get_train_count(self, model_id):
        cursor = self.conn.execute(
            "SELECT train_count FROM models WHERE id = ?", (model_id,)
        )
        result = cursor.fetchone()
        return result[0] if result else 0

    def save_log(self, timestamp, level, message):
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

    def get_recordings_list(self):
        with self.conn:
            cursor = self.conn.execute(
                "SELECT id, model_id, timestamp, path FROM recordings ORDER BY id DESC"
            )
            return cursor.fetchall()

    def get_models(self):
        with self.conn:
            cursor = self.conn.execute("SELECT id FROM models")
            return cursor.fetchall()

    def close(self):
        self.conn.close()

    def save_recording(self, model_id, timestamp, path):
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO recordings (model_id, timestamp, path)
                VALUES (?, ?, ?)
            """,
                (model_id, timestamp, path),
            )

    def save_results(self, model_id, timestamp, x_positions, y_positions, time, died):
        x_positions = json.dumps({"x_positions": x_positions})
        y_positions = json.dumps({"y_positions": y_positions})
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO results (model_id, timestamp, x_positions, y_positions, time, died)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (model_id, timestamp, x_positions, y_positions, time, died),
            )

    def get_results_list(self):
        with self.conn:
            cursor = self.conn.execute(
                "SELECT id, model_id, timestamp, time, died FROM results ORDER BY id DESC",
            )
            return cursor.fetchall()

    def get_results(self, results_id):
        with self.conn:
            cursor = self.conn.execute(
                "SELECT id, model_id, timestamp, x_positions, y_positions, time, died FROM results WHERE id is ?",
                (results_id,),
            )
            return cursor.fetchall()

    def set_highscore(self, model_id, x, score, time):
        """
        Updates highscore if any metric is better than current record
        Returns: bool indicating if any record was beaten
        """
        current = self.get_highscore(model_id)
        if not current:
            # No existing record, insert new one
            with self.conn:
                self.conn.execute(
                    "INSERT INTO highscores (model_id, timestamp, x, score, time) VALUES (?, datetime('now'), ?, ?, ?)",
                    (model_id, x, score, time),
                )
            return True

        _, current_x, current_score, current_time = current
        record_beaten = False

        # Check if any metric is better
        if (
            x > current_x or score > current_score or (time < current_time and time > 0)
        ):  # Ignore 0 time
            with self.conn:
                self.conn.execute(
                    "UPDATE highscores SET timestamp=datetime('now'), x=?, score=?, time=? WHERE model_id=?",
                    (
                        max(x, current_x),
                        max(score, current_score),
                        min(time, current_time),
                        model_id,
                    ),
                )
            record_beaten = True

        return record_beaten

    def get_highscore(self, model_id) -> (int, int, int, str):
        with self.conn:
            cursor = self.conn.execute(
                "SELECT timestamp, x, score, time FROM highscores WHERE model_id is ?",
                (model_id,),
            )
            return cursor.fetchone()
        return (0, 0, 1000000, "NO")
