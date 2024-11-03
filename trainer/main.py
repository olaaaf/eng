import asyncio
import logging
import threading

import torch
from fastapi import FastAPI

from game.runner import Runner
from train.model import SimpleModel
from train.trainer import Trainer
from util.db_handler import DBHandler
from util.logger import setup_logger
from util.routes import app

# app = FastAPI()

# app.include_router(router)

global logger
logger: logging.Logger


@app.on_event("startup")
async def startup_event():
    db = DBHandler()
    logger = setup_logger(db)
    logger.info("Server started")
    db.init_logger(logger)

    # this should be in a loop, just an api call that will start another thread
    # api takes in model_id
    # /train_start and /train_stop
    # model, optimizer = db.load_model(model_id)
    # runner = Runner()
    # trainer = Trainer(0, runner, model, optimizer, db, logger)
    # await trainer.evaluate()


# Dictionary to track running threads by model_id
active_threads = {}


@app.post("/train_start/{model_id}")
async def train_start(model_id: int):
    db = DBHandler()
    logger = setup_logger(db)
    logger.info(f"Training for model {model_id} started")
    if model_id in active_threads:
        return {"status": "Training already in progress for this model."}

    # Load model and optimizer from the database
    _, model, optimizer = db.load_model(model_id)
    if not model:
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        db.save_model(model_id, model, optimizer)
    runner = Runner()
    trainer = Trainer(model_id, runner, model, optimizer, db, logger)

    async def training_loop():
        while True:
            await trainer.evaluate()
            await trainer.train()

    # Start training in a new thread
    training_thread = threading.Thread(target=asyncio.run, args=(training_loop(),))
    training_thread.start()

    # Track the thread
    active_threads[model_id] = training_thread

    return {"status": f"Training started for model {model_id}."}


@app.post("/train_stop/{model_id}")
def train_stop(model_id: int):
    if model_id not in active_threads:
        return {"status": "No training in progress for this model."}

    # Implement a method to signal trainer to stop if needed
    # Currently, FastAPI doesn't stop threads directly, this would require
    # a mechanism within `Trainer` to check for a stop signal.

    training_thread = active_threads.pop(model_id)
    # Here, you would set a flag in Trainer to gracefully stop the loop.
    # Example (pseudocode): `trainer.stop_training = True`

    db = DBHandler()
    logger = setup_logger(db)
    logger.info(f"Training for model {model_id} stopped")
    return {"status": f"Training stopped for model {model_id}."}


@app.on_event("shutdown")
async def shutdown_event():
    db = DBHandler()
    db.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
