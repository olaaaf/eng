import logging

# from game.runner import Runner
# from train.trainer import Trainer
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

    # model, optimizer = db.load_model(0)
    # runner = Runner()
    # trainer = Trainer(0, runner, model, optimizer, db, logger)
    # await trainer.evaluate()


@app.on_event("shutdown")
async def shutdown_event():
    db = DBHandler()
    db.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
