import array
from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel
from typing import List
from db_handler import DBHandler
from trainer import Trainer
from logger import setup_logger
from model import SimpleModel
import torch
import base64
import io

router = APIRouter()

class InputData(BaseModel):
    modelid: int

class ScoreData(BaseModel):
    modelid: int
    frames: int
    position: int 
    dead: bool

class LogData(BaseModel):
    level: str
    message: str

class Steps(BaseModel):
    mario_x: List[int]
    mario_y: List[int]
    mario_x_speed: List[int]
    action:  List[int]

class Episode(BaseModel):
    modelid: int
    final_score: int
    states: Steps

def get_db():
    db = DBHandler()
    try:
        yield db
    finally:
        pass

def get_logger(db: DBHandler = Depends(get_db)):
    return setup_logger(db)

@router.post("/get_model")
async def get_action(model_id, db: DBHandler = Depends(get_db), logger = Depends(get_logger)):
    model, optimizer = db.load_model(model_id)
    if model is None:
        logger.warning(f"Model {model_id} not found. Initializing new model.")
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        db.save_model(model_id, model, optimizer)

    # Serialize the model state_dict to a byte stream
    model_buffer = io.BytesIO()
    
    torch.save(model.state_dict(), model_buffer)
    model_buffer.seek(0)
    
    # Encode the model bytes to base64 so it can be sent in the JSON response
    model_base64 = base64.b64encode(model_buffer.getvalue()).decode('utf-8')
   
    return {"model_base64": model_base64}

@router.post("/submit_episode")
async def submit_score(data: Episode, background_tasks: BackgroundTasks, db: DBHandler = Depends(get_db), logger = Depends(get_logger)):
    logger.info(f"received states: {str(data.states)}")
    
    model, optimizer = db.load_model(data.modelid)
    if model is None:
        logger.error(f"Model {data.modelid} not found for score submission.")
        return {"status": "Error: Model not found"}

    trainer = Trainer(model, optimizer, db, logger)
    for step in data.steps:
        state = torch.tensor([step.mario_x, step.mario_y, step.frame], dtype=torch.float32)
        action = torch.sensor(step.action, dtype=torch.float32)
        trainer.store_step(state, action)
    
    background_tasks.add_task(trainer.train, data.modelid, rewards)
    
    logger.info(f"Score submitted for model {data.modelid}. Training queued.")
    return {"status": "Score submitted and training queued"}

@router.post("/log")
async def log_message(data: LogData, db: DBHandler = Depends(get_db), logger = Depends(get_logger)):
    getattr(logger, data.level.lower())(data.message)
    return {"status": "Log message recorded"}

@router.get("/logs")
async def get_logs(limit: int = 100, db: DBHandler = Depends(get_db)):
    logs = db.get_logs(limit)
    return {"logs": logs}