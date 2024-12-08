from datetime import datetime
from typing import List


class Step:
    def __init__(self):
        self.x_pos = []
        self.y_pos = []
        self.horizontal_speed = []
        self.died = False
        self.time = 0
        self.level = 0
        self.goomba_states: List[bool] = []
        for _ in range(0x001E, 0x0024):
            self.goomba_states.append(False)
        self.goomba_stomps = 0
        self.just_stomped = False
        self.score = []

    def step(self, nes, frame_skip) -> tuple[int, int]:
        self.time += frame_skip

        self.lives = nes[0x75A]
        if self.lives != 2:
            self.died = True

        x_horizontal = nes[0x006D]
        x_on_screen = nes[0x0086]
        horizontal_speed = nes[0x0057]
        # convert to signed integer
        if horizontal_speed > 128:
            horizontal_speed -= 256

        y_position_on_screen = nes[0x00CE]
        x_position = (x_horizontal << 8) | x_on_screen
        score_bcd = [
            nes[0x07DD],  # 1000000 and 100000 place
            nes[0x07DE],  # 10000 and 1000 place
            nes[0x07DF],  # 100 and 10 place
            nes[0x07E0],  # 1 place (if applicable)
            nes[0x07E1],  # 1 place (if applicable)
            nes[0x07E2],  # 1 place (if applicable)
        ]
        # Convert BCD to integer score
        score = 0
        for byte in score_bcd:
            score = score * 100 + ((byte >> 4) * 10) + (byte & 0x0F)

        self.x_pos.append(x_position)
        self.y_pos.append(y_position_on_screen)
        self.score.append(score)
        self.horizontal_speed.append(horizontal_speed)
        self.level = nes[0x0760]
        self.get_enemy_states(nes)

        return (self.lives, self.level)

    def save_to_db(self, model_id, db):
        timestamp = datetime.now().isoformat()
        db.save_results(
            model_id, timestamp, self.x_pos, self.y_pos, self.time, self.died
        )

    def to_tensor(self):
        pass

    def get_enemy_states(self, nes):
        self.just_stomped = False
        for ix, addr in enumerate(range(0x001E, 0x0024)):
            state = nes[addr]
            if state == 0x04 and not self.goomba_states[ix]:  # stomped
                self.goomba_states[ix] = True
                self.goomba_stomps += 1
                self.just_stomped = True
            elif state == 0x00:
                self.goomba_states[ix] = False
