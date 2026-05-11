from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd
import requests, joblib, os

app = FastAPI(title="Blackjack AI")

# Load model


MODEL_PATH = "model/blackjack_ai_model_compressed.pkl"
model = joblib.load(MODEL_PATH)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class BlackjackInput:
    def __init__(self, player_sum: int, dealer_up: int, soft: bool, num_cards: int = 2):
        self.player_sum = player_sum
        self.dealer_up = dealer_up
        self.soft = soft
        self.num_cards = num_cards

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(player_sum: int, dealer_up: int, soft: bool = False, num_cards: int = 2):
    has_ace = 1 if soft or player_sum <= 11 else 0
    
    input_df = pd.DataFrame([[
        player_sum, dealer_up, int(soft), num_cards, has_ace
    ]], columns=['player_sum', 'dealer_up', 'soft', 'num_cards', 'has_ace'])
    
    action = model.predict(input_df)[0]
    confidence = float(model.predict_proba(input_df).max())
    
    return {
        "action": action.upper(),
        "confidence": round(confidence * 100, 1),
        "player_sum": player_sum,
        "dealer_up": dealer_up,
        "soft": soft
      }
