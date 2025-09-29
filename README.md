# Pakistan National AI Cloud & API Gateway - Phase 1

This repository documents the Phase-1 progress of the **Pakistan National AI Cloud & API Gateway** hackathon project.

## Resources
- [AI-Infrastructure (GitHub)](https://github.com/umar178/AI-Infrastructure/tree/Trainer)  
- [Urdu NLP Models (Hugging Face)](https://huggingface.co/umar178/UrduTextClassificationModels/tree/main)  
- [Urdu Multi-Domain Classification Dataset (Hugging Face)](https://huggingface.co/datasets/umar178/UrduMultiDomainClassification)  

---

## Problem Statement
Pakistan lacks a unified AI infrastructure with:
- No standardized data schemas for government datasets  
- Limited availability of NLP models for Urdu/regional languages  
- Lack of reliable and secure compute/API infrastructure  

Phase-1 work addresses these challenges.

---

## Project Deliverables

### 1. Data Schema Design
- JSON/XML schema covering **Health, Education, and Population datasets**  
- Reusable and extensible design for government data exchange  

### 2. Urdu NLP Models
- Trained on **two datasets**:  
  - Public sentiment dataset (limited to sentiment labels)  
  - AI-generated dataset (intent & topic classification)  
- Released a **public dataset** with open license:  
  [Urdu Multi-Domain Classification Dataset](https://huggingface.co/datasets/umar178/UrduMultiDomainClassification)  
- Fine-tuned models published on Hugging Face:  
  [Urdu NLP Models](https://huggingface.co/umar178/UrduTextClassificationModels/tree/main)  
- Achieved **>80% classification accuracy**  

### 3. API Development
Three RESTful APIs deployed on a private VPS with authentication and rate limiting:  
- **`/gettopic`** → Topic classification  
- **`/getintent`** → Intent classification  
- **`/getanalysis`** → Sentiment analysis  

**Request Example:**
```bash
curl -X POST "http://n8n.srv940619.hstgr.cloud:5050/gettopic" ^
  -H "Content-Type: application/json" ^
  -H "Authorization: Bearer <token>" ^
  -d "{"message":"آپ کیسے ہیں"}"
```

**Response Example:**
```json
{
  "label": "Greeting",
  "confidence": 0.92
}
```

---

## Repository Structure
- [AI-Infrastructure (GitHub)](https://github.com/umar178/AI-Infrastructure/tree/Trainer) → Training pipelines & infrastructure  
- [Hugging Face Models](https://huggingface.co/umar178/UrduTextClassificationModels/tree/main) → Fine-tuned Urdu NLP models  
- [Public Dataset](https://huggingface.co/datasets/umar178/UrduMultiDomainClassification) → AI-generated Urdu classification dataset  

---

## Overcoming Challenges
- **Dataset Limitation:** Public datasets only supported sentiment analysis → solved by creating and releasing a custom dataset.  
- **API Security:** Implemented authentication and rate limiting to prevent misuse.  

---

## Conclusion
Phase-1 successfully delivered:
- ✅ Reusable government data schemas  
- ✅ High-performing Urdu NLP models (>80% accuracy)  
- ✅ Secure API endpoints for real-world usage  

This foundation will support **Pakistan’s National AI Cloud & API Gateway** initiative and empower researchers, startups, and government to build localized AI solutions.

---

## License
This project and dataset are released under an **open license** for community use.
