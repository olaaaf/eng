import argparse
import os
import asyncio
import contextlib
import logging
import signal
import sys
import threading
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException

import wandb
from game.runner import Runner
from train.dqn_trainer import DQNTrainer, DQNProfiler
from train.model import SimpleModel
from train.helpers import ConfigFileReward
from util.db_handler import DBHandler
from util.logger import setup_logger
from util.routes import app
import config


@dataclass
class TrainingSession:
    thread: threading.Thread
    trainer: DQNTrainer
    stop_flag: threading.Event


class TrainingManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.active_sessions: Dict[int, TrainingSession] = {}

    @contextlib.contextmanager
    def get_session(self, model_id: int) -> Optional[TrainingSession]:
        """Thread-safe context manager to access training sessions."""
        with self.lock:
            yield self.active_sessions.get(model_id)

    def add_session(self, model_id: int, session: TrainingSession) -> bool:
        """Thread-safe addition of new training session."""
        with self.lock:
            if model_id in self.active_sessions:
                return False
            self.active_sessions[model_id] = session
            return True

    def remove_session(self, model_id: int) -> Optional[TrainingSession]:
        """Thread-safe removal of training session."""
        with self.lock:
            return self.active_sessions.pop(model_id, None)

    def is_active(self, model_id: int) -> bool:
        """Thread-safe check if model is currently training."""
        with self.lock:
            return model_id in self.active_sessions


# Global training manager instance
training_manager = TrainingManager()


@app.on_event("startup")
async def startup_event():
    db = DBHandler()
    logger = setup_logger(db, "server")
    logger.info("Server started")
    db.init_logger(logger)
    # wandb.config


async def training_loop(trainer: DQNTrainer, stop_flag: threading.Event):
    """Training loop that checks for stop signal."""
    trainer.logger.info(f"Started training thread for model {trainer.model_id}")
    while not stop_flag.is_set():
        try:
            await trainer.evaluate()
        except Exception as e:
            trainer.logger.error(f"Error in training loop: {str(e)}")
            break
    trainer.logger.info(f"Training loop ended for model {trainer.model_id}")
    trainer.cleanup()


@app.post("/train_start/{model_id}")
async def train_start(model_id: int):
    if training_manager.is_active(model_id):
        raise HTTPException(
            status_code=400, detail="Training already in progress for this model."
        )

    # Initialize training components
    db = DBHandler()
    logger = setup_logger(db, "trainer_sever")
    epsilon = 1
    episode = 0

    config.create_default(model_id)
    reward_handler = ConfigFileReward(logger, model_id)
    # Load or create model
    try:
        _, model, optimizer, epsilon, episode = db.load_model(model_id, reward_handler)
        if not model:
            model = SimpleModel(reward_handler)
            epsilon = 1
            episode = 0
            optimizer = torch.optim.Adam(model.parameters())
            db.save_model(1, model_id, model, optimizer, episode)
    except Exception as e:
        logger.error(f"Error loading model {model_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to load or create model: {str(e)}"
        )

    # Create training components
    runner = Runner(torch.device("mps" if torch.backends.mps.is_available() else "cpu"))
    # Create a config or load one for the reward system

    trainer = DQNTrainer(
        model_id,
        runner,
        model,
        None,
        # optimizer,
        db,
        epsilon_start=epsilon,
        episode=episode,
        reward_handler=reward_handler,
    )
    stop_flag = threading.Event()

    # Create and start training thread
    async def wrapped_training_loop():
        try:
            await training_loop(trainer, stop_flag)
        finally:
            training_manager.remove_session(model_id)
            logger.info(f"Cleaned up training session for model {model_id}")

    training_thread = threading.Thread(
        target=asyncio.run, args=(wrapped_training_loop(),), name=f"training-{model_id}"
    )

    # Add session to manager
    session = TrainingSession(
        thread=training_thread, trainer=trainer, stop_flag=stop_flag
    )

    if not training_manager.add_session(model_id, session):
        raise HTTPException(
            status_code=400, detail="Training already in progress for this model."
        )

    training_thread.start()

    return {"status": "success", "message": f"Training started for model {model_id}"}


@app.post("/train_stop/{model_id}")
async def train_stop(model_id: int):
    with training_manager.get_session(model_id) as session:
        if not session:
            raise HTTPException(
                status_code=404, detail="No training in progress for this model."
            )

        # Signal the training loop to stop
        session.stop_flag.set()

        # Wait for a short time for the thread to clean up
        session.thread.join(timeout=5.0)

        # Force cleanup if thread hasn't finished
        if session.thread.is_alive():
            training_manager.remove_session(model_id)
            session.trainer.logger.warning(
                f"Force stopped training for model {model_id}"
            )

    return {"status": "success", "message": f"Training stopped for model {model_id}"}


@app.get("/training_status/{model_id}")
async def training_status(model_id: int):
    """Check if a model is currently training."""
    is_active = training_manager.is_active(model_id)
    return {"model_id": model_id, "is_training": is_active}


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup all training sessions on shutdown."""
    with training_manager.lock:
        active_sessions = list(training_manager.active_sessions.items())

    for model_id, session in active_sessions:
        session.stop_flag.set()
        session.thread.join(timeout=2.0)

    db = DBHandler()
    db.close()


def signal_handler(logger):
    logger.info("\nShutting down gracefully...")
    sys.exit(0)


def cli_main(model_id: int):
    """Terminal mode training loop"""
    db = DBHandler()
    logger = setup_logger(db, "cli")
    db.logger = logger
    episode = 0

    def bind_signal(sig, frame):
        signal_handler(logger)

    signal.signal(signal.SIGINT, bind_signal)
    config.create_default(model_id)
    reward_handler = ConfigFileReward(logger, model_id)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    runner = Runner(device)

    trainer: DQNTrainer
    try:
        _, model, optimizer, epsilon, episode = db.load_model(model_id, reward_handler)
        if not model:
            model = SimpleModel(reward_handler)
            optimizer = None
            epsilon = 1
            episode = 0

            # db.save_model(1, model_id, model, None, episode)
    except Exception as e:
        logger.error(f"Error loading model {model_id}: {str(e)}")
        sys.exit(1)

    trainer = DQNTrainer(
        model_id,
        runner,
        model,
        optimizer,
        db,
        reward_handler,
        epsilon_start=epsilon,
        episode=episode,
    )

    trainer.save_model_checkpoint()
    logger.info(f"Starting training for model {model_id}")
    try:
        while True:
            trainer.evaluate()
    except KeyboardInterrupt:
        trainer.cleanup()
        logger.info("Training stopped by user")
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        trainer.cleanup()
        sys.exit(1)


def profiler(model_id: int):
    """Terminal mode training loop"""
    db = DBHandler()
    logger = setup_logger(db, "cli")
    db.logger = logger
    episode = 0

    def bind_signal(sig, frame):
        signal_handler(logger)

    signal.signal(signal.SIGINT, bind_signal)
    config.create_default(model_id)
    reward_handler = ConfigFileReward(logger, model_id)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    runner = Runner(device)

    trainer: DQNTrainer
    try:
        _, model, optimizer, epsilon, episode = db.load_model(model_id, reward_handler)
        if not model:
            model = SimpleModel(reward_handler)
            optimizer = None
            epsilon = 1
            episode = 0

            # db.save_model(1, model_id, model, None, episode)
    except Exception as e:
        logger.error(f"Error loading model {model_id}: {str(e)}")
        sys.exit(1)

    trainer = DQNTrainer(
        model_id,
        runner,
        model,
        optimizer,
        db,
        reward_handler,
        epsilon_start=epsilon,
        episode=episode,
    )
    profiler = DQNProfiler(trainer)
    logger.info(f"Starting training for model {model_id}")
    try:
        profiler.profile_evaluate()
    except KeyboardInterrupt:
        trainer.cleanup()
        logger.info("Training stopped by user")
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        trainer.cleanup()
        sys.exit(1)


def runner(model_id: int):
    """Terminal mode training loop for testing model versions"""
    db = DBHandler()
    logger = setup_logger(db, "cli")
    db.logger = logger

    def bind_signal(sig, frame):
        signal_handler(logger)

    signal.signal(signal.SIGINT, bind_signal)
    config.create_default(model_id)
    reward_handler = ConfigFileReward(logger, model_id)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # Get version range
    start_version = int(input("Start version: "))
    end_version = int(input("End version: "))

    if not os.path.exists("recordings"):
        os.mkdir("recordings")

    # Test each version in range
    for version in range(start_version, end_version + 1):
        logger.info(f"Testing version {version}")
        
        runner = Runner(
            device, 
            record=True, 
            video_save_path="recordings", 
            video_prefix=f"{model_id}_v{version}"
        )
        
        # Initialize wandb
        run = wandb.init(reinit=True)  # Allow multiple runs
        
        # Download the artifact
        artifact = run.use_artifact(
            f"olafercik/mario_shpeed/advanced_model_checkpoint_{model_id}:v{version}",
            type="model",
        )
        artifact_dir = artifact.download()
        
        # Load the model
        filee = os.listdir(artifact_dir)[0]
        checkpoint = torch.load(f"{artifact_dir}/{filee}", map_location=torch.device("cpu"))
        model = SimpleModel(reward_handler)
        model.load_state_dict(checkpoint["model_state_dict"])
        if model is None:
            logger.error(f"Failed to load model version {version}")
            continue
        model.eval()

        trainer = DQNTrainer(
            model_id,
            runner,
            model,
            None,
            db,
            reward_handler,
            epsilon_start=1,
            episode=0,
        )
        
        try:
            trainer.run_only()
        except KeyboardInterrupt:
            logger.info("Training stopped by user")
            break
        except Exception as e:
            logger.error(f"Error testing version {version}: {str(e)}")
            continue
            
        wandb.finish()  # Close current run


def main():
    parser = argparse.ArgumentParser(description="DQN Mario AI Trainer")
    parser.add_argument(
        "-m", "--model", type=int, required=True, help="Model ID to train"
    )
    parser.add_argument(
        "-t", "--terminal", action="store_true", help="Run in terminal mode"
    )
    parser.add_argument("-r", "--run", action="store_true", help="Run only")

    parser.add_argument("-p", "--profiler", action="store_true", help="run profiler")
    args = parser.parse_args()
    if args.run:
        runner(args.model)
    elif args.profiler:
        profiler(args.model)
    elif args.terminal:
        # Setup signal handler for ctrl+c in terminal mode
        cli_main(args.model)
    else:
        # Web mode - use existing FastAPI app
        uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
