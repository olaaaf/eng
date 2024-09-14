from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel
import torch
from typing import List
from db_handler import DBHandler
from trainer import Trainer
from logger import setup_logger
from model import SimpleModel
import base64
import io

router = APIRouter()

class InputData(BaseModel):
    model_id: int

class ScoreData(BaseModel):
    model_id: int
    frames: int
    position: int 
    dead: bool

class LogData(BaseModel):
    level: str
    message: str

class Steps(BaseModel):
    mario_x: [int]
    mario_y: [int]
    mario_x_speed: [int]
    action:  [int]

class Episode(BaseModel):
    model_id: int
    final_score: int
    steps: Steps

def get_db():
    db = DBHandler()
    try:
        yield db
    finally:
        pass

def get_logger(db: DBHandler = Depends(get_db)):
    return setup_logger(db)

@router.post("/get_model")
async def get_action(data: InputData, db: DBHandler = Depends(get_db), logger = Depends(get_logger)):
    model, optimizer = db.load_model(data.model_id)
    if model is None:
        logger.warning(f"Model {data.model_id} not found. Initializing new model.")
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        db.save_model(data.model_id, model, optimizer)

    # Serialize the model state_dict to a byte stream
    model_buffer = io.BytesIO()
    torch.save(model.state_dict(), model_buffer)
    model_buffer.seek(0)
    
    # Encode the model bytes to base64 so it can be sent in the JSON response
    model_base64 = base64.b64encode(model_buffer.getvalue()).decode('utf-8')
   
    return {"model_base64": model_base64}

@router.post("/submit_episode")
async def submit_score(data: Episode, background_tasks: BackgroundTasks, db: DBHandler = Depends(get_db), logger = Depends(get_logger)):
    
    model, optimizer = db.load_model(data.model_id)
    if model is None:
        logger.error(f"Model {data.model_id} not found for score submission.")
        return {"status": "Error: Model not found"}

    trainer = Trainer(model, optimizer, db, logger)
    for step in data.steps:
        state = torch.tensor([step.mario_x, step.mario_y, step.frame], dtype=torch.float32)
        action = torch.sensor(step.action, dtype=torch.float32)
        trainer.store_step(state, action)
    
    background_tasks.add_task(trainer.train, data.model_id, rewards)
    
    logger.info(f"Score submitted for model {data.model_id}. Training queued.")
    return {"status": "Score submitted and training queued"}

@router.post("/log")
async def log_message(data: LogData, db: DBHandler = Depends(get_db), logger = Depends(get_logger)):
    getattr(logger, data.level.lower())(data.message)
    return {"status": "Log message recorded"}

@router.get("/logs")
async def get_logs(limit: int = 100, db: DBHandler = Depends(get_db)):
    logs = db.get_logs(limit)
    return {"logs": logs}