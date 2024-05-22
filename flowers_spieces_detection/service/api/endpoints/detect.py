from fastapi import APIRouter, UploadFile, HTTPException
from PIL import Image
from io import BytesIO
import numpy as np
from service.core.logic.onnx_inference import flower_detector
from service.core.schemas.output import APIOutput

detect_router = APIRouter()

@detect_router.post("/detect", response_model = APIOutput) 
def detect(im: UploadFile):
    if im.filename.split(".")[-1] in ("jpg", "jpeg", "png"):
        image = Image.open(BytesIO(im.file.read()))
        image = np.array(image)
    else:
        raise HTTPException(status_code=415, detail="Not image")

    return flower_detector(image)