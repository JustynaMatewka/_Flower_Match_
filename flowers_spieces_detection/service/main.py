from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from io import BytesIO
import numpy as np
import base64
from service.core.logic.onnx_inference import flower_detector
from service.core.schemas.output import APIOutput

app = FastAPI(project_name="Flower species detection")

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@app.get("/index/", response_class=HTMLResponse)
def root(request: Request):
    context = {'request': request}
    return templates.TemplateResponse("index.html", context)


@app.post("/detect/", response_class=HTMLResponse)
async def detect(request: Request, flowerImage: UploadFile = File(...)):
    contents = await flowerImage.read()
    image = Image.open(BytesIO(contents)).convert('RGB')
    image_np = np.array(image)

    result = flower_detector(image_np)

    image.save(BytesIO(), format="PNG")
    image_base64 = base64.b64encode(BytesIO().getvalue()).decode('utf-8')

    context = {
        "request": request,
        "result": result["emotion"],
        "image": image_base64
    }
    return templates.TemplateResponse("index.html", context)