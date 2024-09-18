from fastapi import FastAPI
from routes import router
from db_handler import DBHandler
from logger import setup_logger

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