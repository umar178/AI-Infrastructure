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

repo_root = Path(__file__).resolve().parent

# region Model v1 Sentiment Analysis variables
model_v1_path = repo_root / "Model 1 - Public dataset" / "urdu_Sentiment_analysis_model"
tokenizer_v1 = AutoTokenizer.from_pretrained(model_v1_path)
model_v1 = AutoModelForSequenceClassification.from_pretrained(model_v1_path)
pipeline_v1 = TextClassificationPipeline(model=model_v1, tokenizer=tokenizer_v1, device=device)
# endregion

# region Model v2 Topic Analysis variables
model_v2_topic_path = repo_root / "Model 2 - Synthetic dataset" / "urdu_Topic_classification_model"
config_topic = AutoConfig.from_pretrained(model_v2_topic_path)
id2label = {}
label2id = {}
with open(model_v2_topic_path / "label_mapping.txt", "r", encoding="utf-8") as f:
    for line in f:
        idx, label = line.strip().split("\t")
        idx = int(idx)
        id2label[idx] = label
        label2id[label] = idx

config_topic.id2label = id2label
config_topic.label2id = label2id
tokenizer_topic_v2 = AutoTokenizer.from_pretrained(model_v2_topic_path)
model_topic_v2 = AutoModelForSequenceClassification.from_pretrained(model_v2_topic_path, config=config_topic)
pipeline_topic_v2 = TextClassificationPipeline(model=model_topic_v2, tokenizer=tokenizer_topic_v2, device=device)
# endregion

# region Model v2 Topic Analysis variables
model_v2_intent_path = repo_root / "Model 2 - Synthetic dataset" / "urdu_Intent_classification_model"
config_intent = AutoConfig.from_pretrained(model_v2_intent_path)
id2label = {}
label2id = {}
with open(model_v2_intent_path / "label_mapping.txt", "r", encoding="utf-8") as f:
    for line in f:
        idx, label = line.strip().split("\t")
        idx = int(idx)
        id2label[idx] = label
        label2id[label] = idx

config_intent.id2label = id2label
config_intent.label2id = label2id
tokenizer_intent_v2 = AutoTokenizer.from_pretrained(model_v2_intent_path)
model_intent_v2 = AutoModelForSequenceClassification.from_pretrained(model_v2_intent_path, config=config_intent)
pipeline_intent_v2 = TextClassificationPipeline(model=model_intent_v2, tokenizer=tokenizer_intent_v2, device=device)
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
    results = pipeline_topic_v2(msg, truncation=True)

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

def classify_message_intent(msg):
    results = pipeline_intent_v2(msg, truncation=True)

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

@app.post("/getIntent")
def get_analysis(request: MessageRequest, authorized: bool = Depends(verify_token)):
    result = classify_message_intent(request.message)
    return JSONResponse(content=result)

if __name__ == "__main__":
    uvicorn.run("chatbot:app", host="0.0.0.0", port=5050)
