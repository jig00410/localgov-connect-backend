import torch
# UPDATED IMPORTS to use the correct BERT classes
from transformers import BertTokenizerFast, BertForSequenceClassification, DistilBertTokenizerFast, DistilBertForSequenceClassification
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO

# Import keyword logic
from keywords_rules import get_category_by_keyword, get_severity_by_keywords

# -------------------------------
# Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] Using device:", device)

# -------------------------------
# Load Text Severity Model (CORRECTED SECTION)
# -------------------------------
# The path to your locally saved and fine-tuned model
severity_model_path = "./severity_model_improved"

# Use the correct BERT classes to load your saved BERT model
severity_tokenizer = BertTokenizerFast.from_pretrained(severity_model_path)
severity_model = BertForSequenceClassification.from_pretrained(severity_model_path)

severity_model.to(device)
severity_model.eval()

severity_labels = ["emergency", "critical", "urgent", "needs attention", "minor"]

# -------------------------------
# Load Text Category Model (No changes here)
# -------------------------------
category_model_path = "./category_model.pth"
category_tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
category_model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=8
)
category_model.load_state_dict(torch.load(category_model_path, map_location=device))
category_model.to(device)
category_model.eval()

category_labels = ["accident", "water", "traffic", "roads", "nature", "fire", "electricity", "sanitation"]

# -------------------------------
# Load Image Model (No changes here)
# -------------------------------
image_model_path = "image_severity_model.pth"
image_model = models.resnet18(pretrained=False)
image_model.fc = torch.nn.Linear(image_model.fc.in_features, 3)
image_model.load_state_dict(torch.load(image_model_path, map_location=device))
image_model.to(device)
image_model.eval()

image_labels = ["fire", "accident", "traffic"]

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------------------
# Functions (No changes here)
# -------------------------------
def classify_text(text):
    # Severity
    inputs = severity_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = severity_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        severity_idx = torch.argmax(probs, dim=1).item()
        severity_conf = probs[0, severity_idx].item()
        severity = severity_labels[severity_idx]

    # Category
    inputs_cat = category_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs_cat = category_model(**inputs_cat)
        probs_cat = torch.softmax(outputs_cat.logits, dim=-1)
        category_idx = torch.argmax(probs_cat, dim=1).item()
        category_conf = probs_cat[0, category_idx].item()
        category = category_labels[category_idx]

    # Keyword override logic
    kw_category = get_category_by_keyword(text) or category
    kw_severity = get_severity_by_keywords(text, kw_category) or severity

    # Confidences: if keywords matched, set to 1.0 (strong rule)
    final_cat_conf = 1.0 if kw_category != category else category_conf
    final_sev_conf = 1.0 if kw_severity != severity else severity_conf

    return kw_severity, final_sev_conf, kw_category, final_cat_conf

def classify_image(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    img = image_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = image_model(img)
        probs = torch.softmax(outputs, dim=-1)
        idx = torch.argmax(probs, dim=1).item()
        conf = probs[0, idx].item()
        category = image_labels[idx]
    return category, conf

def classify_issue(text=None, image_url=None):
    if not text:
        return {"error": "Text is required for issue classification"}

    severity, sev_conf, category, cat_conf = classify_text(text)
    final_category = category
    final_severity = severity

    # Conditional Image Check
    if severity in ["emergency", "urgent"] and category in ["fire", "accident", "traffic"]:
        if image_url:
            img_cat, img_conf = classify_image(image_url)
            if img_cat.lower() == category.lower():
                final_category = img_cat
                final_severity = severity
            else:
                final_category = category

    return {
        "text_severity": severity,
        "text_severity_conf": sev_conf,
        "text_category": category,
        "text_category_conf": cat_conf,
        "final_category": final_category,
        "final_severity": final_severity
    }