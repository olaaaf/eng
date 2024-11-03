import json
from typing import Generator

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from util.db_handler import DBHandler

app = FastAPI()
app.mount("/static", StaticFiles(directory="util/static"), name="static")
templates = Jinja2Templates(directory="util")


def get_db() -> Generator[DBHandler, None, None]:
    db = DBHandler()
    try:
        yield db
    finally:
        db.close()


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/logs", response_class=HTMLResponse)
async def logs_page(request: Request, db: DBHandler = Depends(get_db)):
    logs = db.get_logs()
    print("hey")
    return templates.TemplateResponse("logs.html", {"request": request, "logs": logs})


@app.get("/logs/{log_id}", response_class=HTMLResponse)
async def log_detail(request: Request, log_id: int, db: DBHandler = Depends(get_db)):
    log = db.get_logs(limit=1)  # You'll need to modify DBHandler to get specific log
    if not log:
        raise HTTPException(status_code=404, detail="Log not found")
    return templates.TemplateResponse(
        "log_detail.html", {"request": request, "log": log[0]}
    )


@app.get("/recordings", response_class=HTMLResponse)
async def recordings_page(request: Request, db: DBHandler = Depends(get_db)):
    recordings = db.get_recordings_list()  # You might want to filter by model_id
    return templates.TemplateResponse(
        "recordings.html", {"request": request, "recordings": recordings}
    )


@app.get("/recordings/{recording_id}", response_class=HTMLResponse)
async def recording_detail(
    request: Request, recording_id: int, db: DBHandler = Depends(get_db)
):
    recording = db.get_recordings_list(recording_id)  # You'll need to modify DBHandler
    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")
    return templates.TemplateResponse(
        "recording_detail.html", {"request": request, "recording": recording[0]}
    )


@app.get("/results", response_class=HTMLResponse)
async def results_page(request: Request, db: DBHandler = Depends(get_db)):
    results = db.get_results_list()  # You might want to filter by model_id
    return templates.TemplateResponse(
        "results.html", {"request": request, "results": results}
    )


@app.get("/results/{result_id}", response_class=HTMLResponse)
async def result_detail(
    request: Request, result_id: int, db: DBHandler = Depends(get_db)
):
    result = db.get_results(result_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")

    # Parse the JSON strings back into lists
    result_data = result[0]
    x_positions = json.loads(result_data[3])["x_positions"]
    y_positions = json.loads(result_data[4])["y_positions"]

    return templates.TemplateResponse(
        "result_detail.html",
        {
            "request": request,
            "result": result_data,
            "x_positions": x_positions,
            "y_positions": y_positions,
        },
    )
