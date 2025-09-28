import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# ------------------------------
# 1. Paths
# ------------------------------
MODEL_DIR = "Model 2 - Synthetic dataset/urdu_Topic_classification_model"  # same as your training output_dir
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_mapping.txt")

# ------------------------------
# 2. Load model and tokenizer
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()

# ------------------------------
# 3. Load label mapping
# ------------------------------
id2label = {}
with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    for line in f:
        idx, label = line.strip().split("\t")
        id2label[int(idx)] = label

# ------------------------------
# 4. Prediction function
# ------------------------------
def predict(texts):
    """
    texts: str or list of str (Urdu text)
    returns: list of dicts with label & confidence
    """
    if isinstance(texts, str):
        texts = [texts]

    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)

    predictions = []
    for i, prob in enumerate(probs):
        pred_id = prob.argmax().item()
        predictions.append({
            "text": texts[i],
            "predicted_label": id2label[pred_id],
            "confidence": round(prob[pred_id].item(), 4)
        })
    return predictions

# ------------------------------
# 5. Example usage
# ------------------------------
if __name__ == "__main__":
    examples = [
        # Population
        "پاکستان کی آبادی تیزی سے بڑھ رہی ہے۔",
        "آبادی میں اضافہ معاشی مسائل پیدا کر رہا ہے۔",
        "دیہات سے شہروں کی طرف ہجرت میں اضافہ ہوا ہے۔",
        "آبادی کے دباؤ کی وجہ سے وسائل کم پڑ رہے ہیں۔",
        "پاکستان کی آبادی کتنی ہے۔",
        "پاکستان میں کتنے ہسپتال کتنی آبادی ہے۔",

        # Other
        "کل موسم بہت خوشگوار تھا۔",
        "مجھے کتابیں پڑھنے کا شوق ہے۔",
        "نئی فلم سینما گھروں میں ریلیز ہو گئی ہے۔",
        "کل شہر میں ثقافتی میلہ منعقد ہوا۔",

        # Health
        "ہسپتال میں مریضوں کی تعداد بڑھ گئی ہے۔",
        "صحت مند خوراک انسانی زندگی کے لیے ضروری ہے۔",
        "ڈاکٹروں نے احتیاطی تدابیر اختیار کرنے کی ہدایت دی۔",
        "بچوں کے ٹیکہ جات مہم کا آغاز کیا گیا ہے۔",
        "مجھے ہسپتال کا ایڈریس بتائیں۔",

        # Education
        "تعلیم ہر بچے کا بنیادی حق ہے۔",
        "یونیورسٹی میں داخلے کے لیے امتحانات جاری ہیں۔",
        "اساتذہ طلبہ کی رہنمائی کے لیے ورکشاپ کا انعقاد کر رہے ہیں۔",
        "سکولوں میں نصاب میں تبدیلیاں کی جا رہی ہیں۔",
        "آبادی کا کتنا حصہ تعلیمی ادارے میں پڑھتا ہے۔"
    ]

    results = predict(examples)
    for r in results:
        print(r)