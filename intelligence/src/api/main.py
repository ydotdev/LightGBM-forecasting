"""Standalone forecasting service; no Supabase or main-backend dependency."""
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional
import os
import re
import sys
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from google import genai

_intelligence_dir = Path(__file__).resolve().parents[2]
if str(_intelligence_dir) not in sys.path:
    sys.path.insert(0, str(_intelligence_dir))

# Load .env file
_env_path = _intelligence_dir / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
else:
    load_dotenv()

try:
    from ..ml.predict import DEFAULT_MODEL, load_artifact, predict_demand
except (ImportError, ValueError):
    from src.ml.predict import DEFAULT_MODEL, load_artifact, predict_demand


# ---------- Gemini config ----------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-1.5-flash"


# ---------- Forecasting schemas ----------
class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hospital_id: str = Field(min_length=1)
    medicine_id: str = Field(min_length=1)
    horizon_days: Literal[7, 14] = 7


class DailyPrediction(BaseModel):
    date: date
    predicted_demand: float = Field(ge=0, allow_inf_nan=False)


class PredictionResponse(BaseModel):
    hospital_id: str
    medicine_id: str
    model_version: str
    horizon_days: Literal[7, 14]
    predictions: list[DailyPrediction]


class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    model_version: str
    training_date_range: dict[str, str]
    supported_hospitals: list[str]
    supported_medicines: list[str]
    supported_combinations: list[dict[str, str]]
    feature_names: list[str]
    dataset_origin: str
    evaluation_metrics: dict


# ---------- LLM schemas ----------
class LLMRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(..., min_length=1, max_length=500)


class BrandInfo(BaseModel):
    brand_name: str
    manufacturer: Optional[str] = None


class LLMResponse(BaseModel):
    status: Literal["ok", "out_of_scope", "not_found"]
    medicine: Optional[str] = None
    active_ingredient: Optional[str] = None
    brands: Optional[list[BrandInfo]] = None
    message: Optional[str] = None


# ---------- Static medicine reference DB ----------
STATIC_MEDICINE_DB: dict[str, list[dict]] = {
    "paracetamol": [
        {"brand_name": "Crocin", "manufacturer": "GSK"},
        {"brand_name": "Dolo 650", "manufacturer": "Micro Labs"},
        {"brand_name": "Calpol", "manufacturer": "GSK"},
        {"brand_name": "Tylenol", "manufacturer": "J&J"},
    ],
    "acetaminophen": [
        {"brand_name": "Tylenol", "manufacturer": "J&J"},
        {"brand_name": "Crocin", "manufacturer": "GSK"},
    ],
    "ibuprofen": [
        {"brand_name": "Brufen", "manufacturer": "Abbott"},
        {"brand_name": "Advil", "manufacturer": "Pfizer"},
        {"brand_name": "Motrin", "manufacturer": "J&J"},
    ],
    "amoxicillin": [
        {"brand_name": "Mox", "manufacturer": "Ranbaxy"},
        {"brand_name": "Amoxil", "manufacturer": "GSK"},
        {"brand_name": "Novamox", "manufacturer": "Cipla"},
    ],
    "cetirizine": [
        {"brand_name": "Zyrtec", "manufacturer": "UCB"},
        {"brand_name": "Cetzine", "manufacturer": "GSK"},
        {"brand_name": "Alerid", "manufacturer": "Cipla"},
    ],
    "omeprazole": [
        {"brand_name": "Prilosec", "manufacturer": "AstraZeneca"},
        {"brand_name": "Losec", "manufacturer": "AstraZeneca"},
        {"brand_name": "Omez", "manufacturer": "Dr. Reddy's"},
    ],
    "metformin": [
        {"brand_name": "Glucophage", "manufacturer": "Merck"},
        {"brand_name": "Glycomet", "manufacturer": "USV"},
        {"brand_name": "Fortamet", "manufacturer": "Shionogi"},
    ],
    "atorvastatin": [
        {"brand_name": "Lipitor", "manufacturer": "Pfizer"},
        {"brand_name": "Atorva", "manufacturer": "Zydus"},
        {"brand_name": "Storvas", "manufacturer": "Ranbaxy"},
    ],
    "aspirin": [
        {"brand_name": "Disprin", "manufacturer": "Reckitt"},
        {"brand_name": "Ecosprin", "manufacturer": "USV"},
        {"brand_name": "Bayer Aspirin", "manufacturer": "Bayer"},
    ],
}

MEDICAL_HINTS = {
    "medicine", "medicines", "drug", "drugs", "tablet", "tablets", "pill", "pills",
    "capsule", "capsules", "syrup", "brand", "brands", "ingredient", "ingredients",
    "salt", "composition", "generic", "prescription", "dose", "dosage",
    "paracetamol", "acetaminophen", "ibuprofen", "amoxicillin", "cetirizine",
    "omeprazole", "metformin", "atorvastatin", "aspirin",
}

OFF_TOPIC_BLOCKLIST = {
    "weather", "stock", "crypto", "bitcoin", "recipe", "joke", "poem", "story",
    "code", "python", "javascript", "sql", "hack", "politics", "election",
    "movie", "song", "lyrics", "game", "football", "cricket",
}


# ---------- LLM helpers ----------
def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _is_clearly_off_topic(query: str) -> bool:
    q = _normalize(query)
    if any(word in q for word in OFF_TOPIC_BLOCKLIST):
        return True
    if not any(word in q for word in MEDICAL_HINTS):
        return True
    return False


def _extract_medicine_with_gemini(query: str) -> Optional[str]:
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY is not configured")

    client = genai.Client(api_key=GEMINI_API_KEY)   
    prompt = (
        "You are a strict information extractor for a medicine lookup system.\n"
        "From the user's message, extract ONLY the medicine name or active ingredient "
        "the user is asking about (e.g., 'paracetamol', 'ibuprofen', 'amoxicillin').\n"
        "Rules:\n"
        "- Reply with ONLY the medicine name in lowercase, no punctuation, no explanation.\n"
        "- If the message is not about a specific medicine/ingredient, reply exactly: NONE\n"
        "- If multiple medicines are mentioned, reply with the first one only.\n\n"
        f"User message: {query}\n"
        "Answer:"
    )

    try:
        try:
            resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        except Exception:
            resp = client.models.generate_content(model="gemini-3.5-flash-lite", contents=prompt)
        text = (resp.text or "").strip().lower()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Gemini extraction failed: {exc}") from exc

    text = re.sub(r"[^a-z0-9\s\-]", "", text).strip()
    if not text or text == "none" or len(text) > 60:
        return None
    return text


def _lookup_static(medicine: str) -> Optional[dict]:
    key = medicine.lower().strip()

    if key in STATIC_MEDICINE_DB:
        return {"ingredient": key, "brands": STATIC_MEDICINE_DB[key]}

    for ingredient, brands in STATIC_MEDICINE_DB.items():
        for b in brands:
            if b["brand_name"].lower() == key:
                return {"ingredient": ingredient, "brands": brands}

    for ingredient, brands in STATIC_MEDICINE_DB.items():
        if key in ingredient or ingredient in key:
            return {"ingredient": ingredient, "brands": brands}

    return None


def create_app(model_path: str | Path = DEFAULT_MODEL) -> FastAPI:
    """Cache one artifact per service instance; restart after replacing the model."""
    service = FastAPI(title="PulseGrid AI — Demand Forecasting V1", version="1.0.0")

    @lru_cache(maxsize=1)
    def get_model() -> dict:
        try:
            return load_artifact(model_path)
        except (FileNotFoundError, ValueError, KeyError, EOFError) as exc:
            raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}") from exc

    @service.get("/health", response_model=HealthResponse)
    def health() -> dict:
        return {"status": "ok"}

    @service.get("/model/info", response_model=ModelInfoResponse)
    def model_info() -> dict:
        saved = get_model()
        return dict(model_version=saved["model_version"], training_date_range=saved["training_date_range"],
            supported_hospitals=saved["categories"]["hospital_id"],
            supported_medicines=saved["categories"]["medicine_id"],
            supported_combinations=saved["history"][["hospital_id", "medicine_id"]].drop_duplicates().to_dict("records"),
            feature_names=saved["feature_names"], dataset_origin=saved["evaluation"]["dataset_origin"],
            evaluation_metrics=saved["evaluation"])

    @service.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> dict:
        saved = get_model()
        try:
            return predict_demand(request.hospital_id, request.medicine_id,
                                  request.horizon_days, artifact=saved)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @service.post("/llm", response_model=LLMResponse)
    def llm(payload: LLMRequest) -> dict:
        query = payload.query.strip()

        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        if _is_clearly_off_topic(query):
            return {
                "status": "out_of_scope",
                "message": (
                    "This assistant only answers questions about medicines and their "
                    "brands/active ingredients. Please ask about a specific medicine."
                ),
            }

        medicine = _extract_medicine_with_gemini(query)
        if not medicine:
            return {
                "status": "out_of_scope",
                "message": (
                    "I couldn't identify a specific medicine in your question. "
                    "Please mention a medicine name or active ingredient."
                ),
            }

        found = _lookup_static(medicine)
        if not found:
            return {
                "status": "not_found",
                "medicine": medicine,
                "message": f"No data found for '{medicine}' in the reference database.",
            }

        brands = [BrandInfo(**b) for b in found["brands"]]

        return {
            "status": "ok",
            "medicine": medicine,
            "active_ingredient": found["ingredient"],
            "brands": brands,
            "message": (
                f"Medicine '{medicine}' has active ingredient "
                f"'{found['ingredient']}'. Found {len(brands)} brand(s)."
            ),
        }

    return service


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
