from fastapi import FastAPI, HTTPException, Request, Depends
from pydantic import BaseModel
from typing import Optional
import torch
import torch.nn as nn
from functools import lru_cache

app = FastAPI(title="ML Model API", version="2.2 - Easy Key")

# --- API Key (can now come from query param) ---
API_KEY = "mysecretkey"

def verify_api_key(request: Request):
    key = request.query_params.get("key")
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")

# --- Model definition ---
class MultiInputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

# --- Load model ---
@lru_cache()
def load_model():
    model = MultiInputModel()
    model.load_state_dict(torch.load("multi_input_model.pth"))
    model.eval()
    return model

model = load_model()

# --- Request/Response schemas ---
class InferenceRequest(BaseModel):
    x: float
    z: float

class InferenceResponse(BaseModel):
    prediction: float

# --- Secured Inference Endpoint (via query param now) ---
@app.post("/predict", response_model=InferenceResponse, dependencies=[Depends(verify_api_key)])
def predict(req: InferenceRequest):
    try:
        input_tensor = torch.tensor([[req.x, req.z]], dtype=torch.float32)
        pred = model(input_tensor)
        return {"prediction": float(pred.item())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
