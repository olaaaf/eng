class Step:
    def __init__(self):
        self.x_pos = []
        self.y_pos = []
        self.horizontal_speed = []
        self.died = False
        self.time = 0

    def step(self, x, y, speed, frame_skip, lives):
        self.time += frame_skip
        self.x_pos += x
        self.y_pos += y
        self.horizontal_speed += speed
        if lives != 2:
            self.died = True
        self.log(x, y, speed)

    def log(self, x, y, speed):
        print(f"[{self.time}]: x: {x}, y: {y}, speed: {speed}")
        if self.died:
            print("DIED")
