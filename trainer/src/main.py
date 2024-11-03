import torch
import torch.nn as nn
from fastapi import FastAPI

from db_handler import DBHandler
from game.runner import Runner
from logger import setup_logger
from routes import router

app = FastAPI()

app.include_router(router)

@app.on_event("startup")
async def startup_event():
    db = DBHandler()
    logger = setup_logger(db)
    logger.info("Server started")

@app.on_event("shutdown")
async def shutdown_event():
    db = DBHandler()
    db.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

class SimpleModel(nn.Module):
    def __init__(self, random_weights=True):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(3840, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 6)
        if random_weights:
            nn.init.xavier_uniform(self.fc1.weight)
            nn.init.xavier_uniform(self.fc2.weight)
            nn.init.xavier_uniform(self.fc3.weight)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return [1 if a > 0.5 else 0 for a in torch.sigmoid(self.fc3(x))]

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    @classmethod
    def load_model(cls, path):
        model = cls()
        model.load_state_dict(torch.load(path))
        return model


r = Runner()
model = SimpleModel()
input = r.next()
print(input.size())
output = model.forward(input)
while not r.step.died:
    input = r.next(output)
    output = model.forward(input)

    print(f"output: {r.controller_to_text(output)}")
