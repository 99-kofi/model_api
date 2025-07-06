from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
from functools import lru_cache

app = FastAPI()

# Model class
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

@lru_cache()
def load_model():
    model = MultiInputModel()
    model.load_state_dict(torch.load("multi_input_model.pth"))
    model.eval()
    return model

model = load_model()

class InferenceRequest(BaseModel):
    x: float
    z: float

class InferenceResponse(BaseModel):
    prediction: float

@app.post("/predict", response_model=InferenceResponse)
def predict(req: InferenceRequest):
    try:
        input_tensor = torch.tensor([[req.x, req.z]], dtype=torch.float32)
        pred = model(input_tensor)
        return {"prediction": float(pred.item())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# For local running
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)