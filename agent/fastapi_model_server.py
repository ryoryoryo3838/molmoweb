import os

# Set before any `olmo` import (lazy in model_backends). Public molmo2 warns if unset.
os.environ.setdefault(
    "MOLMO_DATA_DIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "molmo_data")
)

import queue
import torch
from fastapi import FastAPI
from pydantic import BaseModel

from agent.model_backends import HFActionPredictor, NativeActionPredictor
from utils.vis_utils.image import base64_to_numpy_image


CKPT = os.environ.get("CKPT")
if CKPT is None:
    print("Warning: environment variable CKPT is not set")

NUM_PREDICTORS = int(os.environ.get("NUM_PREDICTORS", "1"))
PREDICTOR_TYPE = os.environ.get("PREDICTOR_TYPE", "native")
DEVICE = os.environ.get("DEVICE")

TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
TOP_P = float(os.environ.get("TOP_P", "0.8"))


def create_predictor_pool(
    ckpt: str | None,
    num_predictors: int = 1,
    predictor_type: str = "native",
    device_override: str | None = None,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.8,
) -> queue.Queue:
    if not ckpt:
        raise ValueError(
            "CKPT is required. Set CKPT env var or pass a checkpoint path to scripts/start_server.sh"
        )

    pool: queue.Queue = queue.Queue(maxsize=num_predictors)

    if device_override:
        devices = [device_override] * num_predictors
        accelerator = f"manual ({device_override})"
    elif torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        devices = [f"cuda:{i % gpu_count}" for i in range(num_predictors)]
        accelerator = f"cuda ({gpu_count} visible GPU(s))"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        devices = ["mps"] * num_predictors
        accelerator = "mps"
    else:
        devices = ["cpu"] * num_predictors
        accelerator = "cpu"

    print(f"Using checkpoint: {ckpt}")
    print(
        f"Accelerator: {accelerator}, predictors: {num_predictors}, type: {predictor_type}"
    )

    for device in devices:
        if predictor_type == "hf":
            predictor = HFActionPredictor(checkpoint=ckpt, device=device)
        elif predictor_type == "native":
            predictor = NativeActionPredictor(
                checkpoint=ckpt,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        else:
            raise ValueError(f"Unknown predictor_type: {predictor_type}")

        print(f"Created {type(predictor).__name__} on {device}")
        pool.put(predictor)

    return pool


predictor_pool = create_predictor_pool(
    ckpt=CKPT,
    num_predictors=NUM_PREDICTORS,
    predictor_type=PREDICTOR_TYPE,
    device_override=DEVICE,
    temperature=TEMPERATURE,
    top_p=TOP_P,
)

app = FastAPI()


class PredictRequest(BaseModel):
    prompt: str
    image_base64: str
    past_actions: list | None = None
    temperature: float | None = None
    top_p: float | None = None


@app.post("/predict")
def predict(request: PredictRequest):
    global predictor_pool
    image_np = base64_to_numpy_image(request.image_base64)

    try:
        predictor = predictor_pool.get(timeout=30)
    except queue.Empty:
        return "Predictor error: All predictors are busy"

    try:
        saved = {
            "temperature": getattr(predictor, "temperature", None),
            "top_p": getattr(predictor, "top_p", None),
        }
        if request.temperature is not None:
            predictor.temperature = request.temperature
        if request.top_p is not None:
            predictor.top_p = request.top_p

        try:
            result = predictor.predict(
                request.prompt, image_np, past_actions=request.past_actions
            )
        except Exception as e:
            return f"Predictor error: {str(e)}"
        finally:
            for k, v in saved.items():
                if v is not None:
                    setattr(predictor, k, v)
    finally:
        predictor_pool.put(predictor)
    return result
