"""
PhotoScan API
- EfficientNet-B0 checkpoint: ../models/pathoscan_efficientnet_b0.pth
- SmolLM2-360M-Instruct: downloaded into ../models/SmolLM2-360M-Instruct
Run from project root: uvicorn backend.app:app --reload
"""
from pathlib import Path
from typing import Optional
import json
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
CHECKPOINT_PATH = MODEL_DIR / "pathoscan_efficientnet_b0.pth"
SMOLLM_DIR = MODEL_DIR / "SmolLM2-360M-Instruct"

app = FastAPI(title="PathoScan API", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_LABELS = {
    "AK": ("Actinic Keratosis", "A rough, scaly lesion category in the ISIC label set."),
    "BCC": ("Basal Cell Carcinoma", "A common type of skin cancer represented in the ISIC label set."),
    "BKL": ("Benign Keratosis-like Lesion", "A group of benign keratosis-like appearances in the ISIC label set."),
    "DF": ("Dermatofibroma", "A usually benign fibrous skin lesion category."),
    "MEL": ("Melanoma", "A potentially serious skin cancer category requiring professional assessment."),
    "NV": ("Melanocytic Nevus", "A melanocytic naevus (mole) category."),
    "SCC": ("Squamous Cell Carcinoma", "A skin cancer category requiring professional assessment."),
    "VASC": ("Vascular Lesion", "A lesion category associated with blood vessels.")
}
CLASS_NAMES = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]
model = None
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# The language model is optional: classification still works if it is not installed.
llm_tokenizer = None
llm_model = None
llm_error = None

def load_classifier():
    global model, CLASS_NAMES, transform
    if not CHECKPOINT_PATH.exists():
        raise RuntimeError(
            f"Classifier checkpoint missing: {CHECKPOINT_PATH}. "
            "Copy pathoscan_efficientnet_b0.pth into the models folder."
        )
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    names = checkpoint.get("class_names", CLASS_NAMES)
    net = models.efficientnet_b0(weights=None)
    net.classifier[1] = nn.Linear(net.classifier[1].in_features, len(names))
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval().to(DEVICE)
    model = net
    CLASS_NAMES = list(names)
    size = int(checkpoint.get("image_size", 224))
    mean = checkpoint.get("mean", [0.485, 0.456, 0.406])
    std = checkpoint.get("std", [0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((size, size)), transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    print(f"EfficientNet-B0 loaded on {DEVICE}; validation accuracy in checkpoint: "
          f"{checkpoint.get('validation_accuracy', 'not recorded')}")

def load_smol():
    global llm_tokenizer, llm_model, llm_error
    if not SMOLLM_DIR.exists() or not any(SMOLLM_DIR.iterdir()):
        llm_error = "SmolLM2 files not found. Run: python download_smol_model.py"
        print(llm_error)
        return
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        llm_tokenizer = AutoTokenizer.from_pretrained(str(SMOLLM_DIR), local_files_only=True)
        # Keep the small language model on CPU to preserve VRAM for the image classifier.
        llm_model = AutoModelForCausalLM.from_pretrained(
            str(SMOLLM_DIR), torch_dtype=torch.float32, local_files_only=True
        ).to("cpu").eval()
        llm_error = None
        print("SmolLM2-360M-Instruct loaded on CPU.")
    except Exception as exc:
        llm_error = f"SmolLM2 could not be loaded: {exc}"
        print(llm_error)

def make_suggestion(code: str, confidence: float) -> str:
    condition = CLASS_LABELS.get(code, (code, "Skin lesion category."))[0]
    fallback = (
        f"The image classifier's top category is {condition}. This is an automated "
        "research result, not a diagnosis. Arrange an assessment with a qualified "
        "healthcare professional, especially if the spot is new, changing, bleeding, "
        "itching, painful, or not healing. Do not start treatment based on this result."
    )
    if llm_model is None:
        return fallback
    prompt = f"""<|im_start|>system
You provide brief, cautious, general skin-health education. You are not a doctor.
Never diagnose, prescribe, recommend medication, or claim the image result is certain.
Do not infer symptoms or visual details that were not supplied. Explain that a clinician
must assess the lesion. Give only safe next steps and warning signs for prompt medical
review. Keep the answer under 100 words.<|im_end|>
<|im_start|>user
An image classifier predicted the ISIC 2019 category {code} ({condition}) with a model
score of {confidence:.1f}%. Explain what this category generally means and give cautious
next steps. Make clear that the score is not a probability of disease and this is not
a diagnosis.<|im_end|>
<|im_start|>assistant
"""
    try:
        inputs = llm_tokenizer(prompt, return_tensors="pt")
        with torch.inference_mode():
            output = llm_model.generate(
                **inputs, max_new_tokens=120, do_sample=False,
                repetition_penalty=1.08, pad_token_id=llm_tokenizer.eos_token_id
            )
        answer = llm_tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        return answer if answer else fallback
    except Exception as exc:
        print("SmolLM generation failed:", exc)
        return fallback

@app.on_event("startup")
def startup():
    load_classifier()
    load_smol()

@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "classifier_not_loaded",
        "classifier": model is not None,
        "suggestion_model": llm_model is not None,
        "suggestion_model_message": llm_error,
        "device": str(DEVICE)
    }

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(503, "Classifier is not loaded. Check the checkpoint path.")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload a valid image.")
    raw = await file.read()
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(413, "Image exceeds the 10 MB limit.")
    try:
        image = Image.open(__import__("io").BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "The uploaded file is not a readable image.")
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu()
    order = torch.argsort(probs, descending=True)[:3]
    top_idx = int(order[0])
    code = CLASS_NAMES[top_idx]
    confidence = float(probs[top_idx]) * 100
    condition, description = CLASS_LABELS.get(code, (code, ""))
    top3 = [
        {"code": CLASS_NAMES[int(i)],
         "condition": CLASS_LABELS.get(CLASS_NAMES[int(i)], (CLASS_NAMES[int(i)], ""))[0],
         "score": round(float(probs[int(i)]) * 100, 2)}
        for i in order
    ]
    # Do not translate a classifier score into clinical severity or urgency.
    suggestion = make_suggestion(code, confidence)
    return {
        "condition": condition, "condition_code": code,
        "confidence": round(confidence, 2), "description": description,
        "top3": top3, "suggestion": suggestion,
        "model_score_note": "Classifier score is not a calibrated probability of disease.",
        "disclaimer": "Research/educational screening only. This result cannot diagnose or rule out a condition. Please consult a qualified healthcare professional."
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
