import io
import json
from typing import Generator

import matplotlib.pyplot as plt
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image

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


def create_overlay_graph(x_positions, y_positions) -> bytes:
    """
    Create a graph overlay on the map image with flipped y coordinates.
    Returns the image as bytes.
    """
    # Load the background image
    background = Image.open("util/static/map.png")

    # Create figure with the same size as the background
    dpi = 100
    figsize = (background.size[0] / dpi, background.size[1] / dpi)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Display the background image
    ax.imshow(background, extent=[0, 3584, 0, 240])

    # Flip y coordinates
    flipped_y = [240 - y for y in y_positions]

    # Plot the path
    ax.plot(x_positions, flipped_y, "r-", linewidth=2, alpha=0.7)

    # Add start and end points
    ax.scatter(
        x_positions[0], flipped_y[0], color="green", s=100, label="Start", zorder=5
    )
    ax.scatter(
        x_positions[-1], flipped_y[-1], color="red", s=100, label="End", zorder=5
    )

    # Configure the plot
    ax.set_xlim(0, 3584)
    ax.set_ylim(0, 240)
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.close(fig)
    buf.seek(0)

    return buf.getvalue()


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


@app.get("/dynamic-graph/{result_id}")
async def dynamic_graph(result_id: int, db: DBHandler = Depends(get_db)):
    """
    Generate and return the dynamic graph overlay image for a specific result
    """
    result = db.get_results(result_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")

    result_data = result[0]
    x_positions = json.loads(result_data[3])["x_positions"]
    y_positions = json.loads(result_data[4])["y_positions"]

    image_bytes = create_overlay_graph(x_positions, y_positions)
    return Response(content=image_bytes, media_type="image/png")
