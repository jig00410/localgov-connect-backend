from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from severity_model import classify_text, classify_image, classify_issue

app = FastAPI(title="Issue Severity & Category API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Request schemas
# -------------------------------
class TextRequest(BaseModel):
    text: str

class ImageRequest(BaseModel):
    image_url: str

class IssueRequest(BaseModel):
    text: str = None
    image_url: str = None

# -------------------------------
# Routes
# -------------------------------
@app.get("/")
def read_root():
    return {"message": "API is running!"}

@app.post("/predict-text")
def predict_text_endpoint(request: TextRequest):
    severity, sev_conf, category, cat_conf = classify_text(request.text)
    return {
        "text": request.text,
        "severity": severity,
        "severity_confidence": sev_conf,
        "category": category,
        "category_confidence": cat_conf
    }

@app.post("/predict-image")
def predict_image_endpoint(request: ImageRequest):
    category, conf = classify_image(request.image_url)
    return {
        "image_url": request.image_url,
        "image_category": category,
        "confidence": conf
    }

@app.post("/predict-issue")
def predict_issue_endpoint(request: IssueRequest):
    if not request.text and not request.image_url:
        return {"error": "Provide at least text or image_url"}
    result = classify_issue(text=request.text, image_url=request.image_url)
    return {"result": result}
