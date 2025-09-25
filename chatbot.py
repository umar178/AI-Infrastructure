from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

class MessageRequest(BaseModel):
    message: str

def analyze_sentiment(message):
    repo_root = Path(__file__).resolve().parent
    model_path = repo_root / "urdu-sentiment-analysis" / "models" / "fine_tuned_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = 0 if torch.cuda.is_available() else -1
    pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=device)
    result = pipeline(message, truncation=True, max_length=512)[0]
    print(f"Message: {message}")
    print(f"Predicted label: {result['label']}, Confidence: {result['score']:.4f}")
    return result

app = FastAPI()

@app.post("/getanalysis")
def get_analysis(request: MessageRequest):
    result = analyze_sentiment(request.message)
    return JSONResponse(content=result)

if __name__ == "__main__":
    uvicorn.run("chatbot:app", host="0.0.0.0", port=5050)
