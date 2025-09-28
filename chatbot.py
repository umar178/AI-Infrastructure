from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import os
from dotenv import load_dotenv
from transformers import AutoConfig

load_dotenv()
device = 0 if torch.cuda.is_available() else -1

# Security setup
security = HTTPBearer()
API_TOKEN = os.getenv("API_KEY")

# region Root, model paths and config
repo_root = Path(__file__).resolve().parent
model_v2_path = repo_root / "Model 2 - Synthetic dataset" / "urdu_Topic_classification_model"
config = AutoConfig.from_pretrained(model_v2_path)
id2label = {}
label2id = {}
with open(model_v2_path / "label_mapping.txt", "r", encoding="utf-8") as f:
    for line in f:
        idx, label = line.strip().split("\t")
        idx = int(idx)
        id2label[idx] = label
        label2id[label] = idx

config.id2label = id2label
config.label2id = label2id
# endregion

# region Model v1 Sentiment Analysis variables
model_v1_path = repo_root / "Model 1 - Public dataset" / "urdu_Sentiment_analysis_model"
tokenizer_v1 = AutoTokenizer.from_pretrained(model_v1_path)
model_v1 = AutoModelForSequenceClassification.from_pretrained(model_v1_path)
pipeline_v1 = TextClassificationPipeline(model=model_v1, tokenizer=tokenizer_v1, device=device)
# endregion

# region Model v2 Topic Analysis variables
tokenizer_v2 = AutoTokenizer.from_pretrained(model_v2_path)
model_v2 = AutoModelForSequenceClassification.from_pretrained(model_v2_path, config=config)
pipeline_v2 = TextClassificationPipeline(model=model_v2, tokenizer=tokenizer_v2, device=device)
# endregion

class MessageRequest(BaseModel):
    message: str

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True

def analyze_sentiment(message):
    result = pipeline_v1(message, truncation=True, max_length=512)[0]
    print(f"Message: {message}")
    print(f"Predicted label: {result['label']}, Confidence: {result['score']:.4f}")
    return result


def classify_message_topic(msg):
    results = pipeline_v2(msg, truncation=True)

    if isinstance(msg, str):
        result = results[0]
        return {
            "text": msg,
            "predicted_label": result["label"],
            "confidence": round(result["score"], 4)
        }
    else:
        output = []
        for i, r in enumerate(results):
            output.append({
                "text": msg[i],
                "predicted_label": r["label"],
                "confidence": round(r["score"], 4)
            })
        return output

app = FastAPI()

@app.post("/getanalysis")
def get_analysis(request: MessageRequest, authorized: bool = Depends(verify_token)):
    result = analyze_sentiment(request.message)
    return JSONResponse(content=result)

@app.post("/getTopic")
def get_analysis(request: MessageRequest, authorized: bool = Depends(verify_token)):
    result = classify_message_topic(request.message)
    return JSONResponse(content=result)

if __name__ == "__main__":
    uvicorn.run("chatbot:app", host="0.0.0.0", port=5050)
