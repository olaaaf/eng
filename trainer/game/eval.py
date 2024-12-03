from datetime import datetime


class Step:
    def __init__(self):
        self.x_pos = []
        self.y_pos = []
        self.horizontal_speed = []
        self.died = False
        self.time = 0

    def step(self, x, y, speed, frame_skip, lives):
        self.time += frame_skip
        self.x_pos.append(x)
        self.y_pos.append(y)
        self.horizontal_speed.append(speed)
        if lives != 2:
            self.died = True

    def save_to_db(self, model_id, db):
        timestamp = datetime.now().isoformat()
        db.save_results(
            model_id, timestamp, self.x_pos, self.y_pos, self.time, self.died
        )

    def to_tensor(self):
        pass