import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['USE_TF'] = '0'
os.environ['TRANSFORMERS_NO_TF'] = '1'

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import threading
import traceback
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import difflib
import math

app = FastAPI(title="ABSA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Global model storage
MODEL = {
    "classifier": None,
    "loaded": False,
    "load_error": None,
    "loading": False
}

class AnalysisRequest(BaseModel):
    review: str
    aspects: str


def map_label(label: str) -> str:
    """Map model labels to standard sentiment labels"""
    s = label.lower()
    if "pos" in s:
        return "Positive"
    if "neg" in s:
        return "Negative"
    return "Neutral"


def parse_output(raw: Any) -> Dict[str, float]:
    """Parse model output and extract sentiment probabilities"""
    # Pipeline output can be either a list of score-dicts or a nested list
    if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], list):
        raw = raw[0]

    probs = {"Positive": 0.0, "Neutral": 0.0, "Negative": 0.0}

    # Handle nlptown 5-star style outputs (labels like '1 star', '2 stars', ...)
    if isinstance(raw, list) and len(raw) > 0 and any("star" in r.get("label", "").lower() for r in raw):
        # Expect five-class probabilities for 1-5 stars
        star_scores = {int(r["label"].split()[0]): float(r["score"]) for r in raw}
        # Negative: 1-2 stars, Neutral: 3 stars, Positive: 4-5 stars
        probs["Negative"] = star_scores.get(1, 0.0) + star_scores.get(2, 0.0)
        probs["Neutral"] = star_scores.get(3, 0.0)
        probs["Positive"] = star_scores.get(4, 0.0) + star_scores.get(5, 0.0)
    else:
        # Generic handling: map label text to Positive/Neutral/Negative buckets
        for r in raw:
            label = r.get("label", "").lower()
            score = float(r.get("score", 0.0))
            if "pos" in label or label in ("positive", "pos"):
                probs["Positive"] += score
            elif "neg" in label or label in ("negative", "neg"):
                probs["Negative"] += score
            else:
                probs["Neutral"] += score

    total = sum(probs.values()) or 1.0
    return {k: v / total for k, v in probs.items()}


def decide_label(probs: Dict[str, float]) -> str:
    """Decide final sentiment label based on probabilities"""
    if probs["Positive"] > probs["Negative"] + 0.1:
        return "Positive"
    elif probs["Negative"] > probs["Positive"] + 0.1:
        return "Negative"
    return "Neutral"


# --- Advanced heuristics for real-world performance ---
NEGATIONS = {"not", "no", "n't", "never", "none", "nobody", "nothing", "hardly", "scarcely"}
INTENSIFIERS = {
    "extremely": 1.4,
    "very": 1.25,
    "really": 1.15,
    "quite": 1.1,
    "slightly": 0.85,
    "somewhat": 0.9,
}

# Configuration and tuning knobs (change these to tune behavior)
CONFIG = {
    # How many tokens around the aspect to look for negation words
    "NEGATION_WINDOW": 4,
    # Base weight for fuzzy matches when aspect literal not in sentence
    # Final sentence weight = FUZZY_BASE + FUZZY_SCALE * fuzzy_score
    "FUZZY_BASE": 0.3,
    "FUZZY_SCALE": 0.7,
    # Whether to apply recency weighting (later sentences count more)
    "RECENCY_WEIGHT": True,
    # Strength of recency effect (0=no effect, 1=moderate)
    "RECENCY_STRENGTH": 0.5,
    # Intensity multipliers (can be extended)
    "INTENSIFIERS": INTENSIFIERS,
}

# Notes on configuring behavior & tuning:
# - NEGATION_WINDOW: increase to detect negations that are farther from the aspect
#   (may introduce false positives). Default 4 tokens is a reasonable start.
# - INTENSIFIERS: add or adjust words and multipliers based on your domain
#   (e.g., "incredibly": 1.5 for very strong intensifier).
# - FUZZY_BASE/FUZZY_SCALE: control how much non-explicit sentences contribute.
#   If your aspects are often paraphrased, increase FUZZY_BASE. To reduce
#   influence from non-explicit sentences, lower FUZZY_BASE (e.g., 0.1).
# - RECENCY_WEIGHT & RECENCY_STRENGTH: enable to prefer later sentences in multi-
#   sentence reviews (useful when reviewers summarize near the end).
# - For production, consider ensembling multiple models or fine-tuning a domain
#   model; these heuristics are best used to handle edge cases only.


def has_negation_near(aspect: str, sentence: str, window: int | None = None) -> bool:
    """Return True if a negation word appears within `window` tokens of the aspect."""
    if window is None:
        window = CONFIG.get("NEGATION_WINDOW", 4)
    tokens = [t.lower() for t in sentence.split()]
    aspect_tokens = [t.lower() for t in aspect.split()]
    # find approximate index of aspect in tokens
    for i in range(len(tokens)):
        # simple match of aspect first token
        if aspect_tokens and tokens[i].startswith(aspect_tokens[0]):
            # check nearby tokens
            left = max(0, i - window)
            right = min(len(tokens), i + window + 1)
            for t in tokens[left:right]:
                if t in NEGATIONS:
                    return True
    return False


def intensity_multiplier(sentence: str) -> float:
    """Return a multiplier (>0) based on presence of intensity words in the sentence."""
    s = sentence.lower()
    mult = 1.0
    for word, factor in CONFIG.get("INTENSIFIERS", {}).items():
        if word in s:
            mult *= factor
    # cap multiplier to reasonable bounds
    return min(max(mult, 0.6), 1.6)


def fuzzy_aspect_in_sentence(aspect: str, sentence: str) -> float:
    """Return a similarity score (0-1) between aspect and words in sentence using difflib."""
    words = [w.strip('.,!?;()[]') for w in sentence.split()]
    best = 0.0
    for w in words:
        score = difflib.SequenceMatcher(None, aspect.lower(), w.lower()).ratio()
        if score > best:
            best = score
    return best


def load_absa_model():
    """Load the ABSA model"""
    if MODEL["loaded"]:
        return True
    
    if MODEL["loading"]:
        return False
    
    MODEL["loading"] = True
    
    try:
        from transformers import pipeline

        # Use a stable, widely available sentiment model that supports text-pair
        # inference on CPU. The yangheng model may require model class variants
        # that are not available in this Transformers build; switching to a
        # 5-star sentiment model (nlptown) provides reliable outputs we can
        # bucket into Positive/Neutral/Negative for ABSA.
        print("Loading ABSA model (distilbert-base-uncased-finetuned-sst-2-english)...")

        # Use a small, public sentiment model that does not require authentication.
        # Explicitly pass use_auth_token=False to avoid sending any invalid Authorization header
        classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            tokenizer="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1,
            return_all_scores=True,
            use_auth_token=False,
        )

        # Test the model (single-sentence). We'll format inputs in analyze to include aspect context.
        test_result = classifier("The battery is good")
        
        if test_result:
            print("Model loaded successfully!")
            MODEL["classifier"] = classifier
            MODEL["loaded"] = True
            MODEL["loading"] = False
            return True
        else:
            raise ValueError("Model test failed")
            
    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        MODEL["load_error"] = error_msg
        MODEL["loading"] = False
        print(f"Model load error: {error_msg}")
        return False


@app.on_event("startup")
def startup_event():
    """Preload model in background"""
    def preload():
        try:
            load_absa_model()
        except Exception as e:
            print(f"Preload error: {e}")
    
    thread = threading.Thread(target=preload, daemon=True)
    thread.start()


@app.get("/")
async def home():
    """Serve the main HTML page"""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return JSONResponse(
        content={"error": "Frontend not found. Please create static/index.html"},
        status_code=404
    )


@app.get("/status")
async def status():
    """Check model loading status"""
    return JSONResponse(content={
        "model_loaded": MODEL["loaded"],
        "loading": MODEL["loading"],
        "load_error": MODEL["load_error"]
    })


@app.post("/analyze")
async def analyze_absa(request: AnalysisRequest):
    """Analyze review for aspect-based sentiment"""
    import re

    try:
        # Validate input
        review = request.review.strip()
        aspects_raw = request.aspects.strip()

        if not review or not aspects_raw:
            return JSONResponse(content={"error": "Review and aspects are required"}, status_code=400)

        # Load model if not loaded
        if not MODEL["loaded"]:
            if not load_absa_model():
                return JSONResponse(content={"error": "Model failed to load", "details": MODEL["load_error"]}, status_code=500)

        # Parse aspects
        aspects_list = [a.strip() for a in aspects_raw.split(",") if a.strip()]
        if not aspects_list:
            return JSONResponse(content={"error": "No valid aspects provided"}, status_code=400)

        results: List[Dict[str, Any]] = []

        for aspect in aspects_list:
            try:
                # Find sentences containing the aspect; fallback to full review
                sentences = [s.strip() for s in re.split(r'[\.\!\?]', review) if aspect.lower() in s.lower()]
                if not sentences:
                    sentences = [review]

                sentence_probs = []
                for idx, sentence in enumerate(sentences):
                    inp = f"{sentence} (aspect: {aspect})"
                    preds = MODEL["classifier"](inp)
                    probs = parse_output(preds)

                    # Apply heuristics: intensity and negation adjustment
                    mult = intensity_multiplier(sentence)
                    if has_negation_near(aspect, sentence):
                        # If negation near aspect, flip positive/negative influence
                        adj_positive = (probs.get("Neutral", 0.0) + probs.get("Negative", 0.0) * 0.9) * (1.0 / mult)
                        adj_negative = probs.get("Positive", 0.0) * 0.9 * mult
                        # Normalize
                        ssum = adj_positive + adj_negative + probs.get("Neutral", 0.0)
                        if ssum > 0:
                            probs = {
                                "Positive": adj_positive / ssum,
                                "Negative": adj_negative / ssum,
                                "Neutral": probs.get("Neutral", 0.0) / ssum,
                            }
                    else:
                        # Boost positivity/negativity based on intensity
                        probs["Positive"] = min(1.0, probs.get("Positive", 0.0) * mult)
                        probs["Negative"] = min(1.0, probs.get("Negative", 0.0) * mult)

                    # If aspect isn't literally present, use fuzzy match to reduce weight
                    fuzzy = fuzzy_aspect_in_sentence(aspect, sentence)
                    if aspect.lower() not in sentence.lower():
                        weight = CONFIG.get("FUZZY_BASE", 0.3) + CONFIG.get("FUZZY_SCALE", 0.7) * fuzzy
                    else:
                        weight = 1.0

                    # Apply recency weighting if enabled
                    if CONFIG.get("RECENCY_WEIGHT", False) and len(sentences) > 1:
                        recency_strength = CONFIG.get("RECENCY_STRENGTH", 0.5)
                        recency_factor = (idx / max(1, len(sentences) - 1)) * recency_strength
                        weight *= (1.0 + recency_factor)

                    sentence_probs.append((probs, weight))

                # Average probabilities across sentences
                # Weighted average across sentences
                total_w = 0.0
                avg_probs = {"Positive": 0.0, "Neutral": 0.0, "Negative": 0.0}
                for p, w in sentence_probs:
                    total_w += w
                    for k in avg_probs:
                        avg_probs[k] += p.get(k, 0.0) * w
                if total_w > 0:
                    for k in avg_probs:
                        avg_probs[k] /= total_w

                final_score = avg_probs.get("Positive", 0.0)  # 0-1
                sentiment = decide_label(avg_probs)
                confidence = max(avg_probs.values()) * 100

                results.append({
                    "Aspect": aspect,
                    "Sentiment": sentiment,
                    "Probabilities": {
                        "Positive": round(avg_probs["Positive"], 3),
                        "Neutral": round(avg_probs["Neutral"], 3),
                        "Negative": round(avg_probs["Negative"], 3),
                    },
                    "Score (1-10)": round(final_score * 10, 2),
                    "Confidence (%)": round(confidence, 1),
                    "Details": {
                        "negation_window": CONFIG.get("NEGATION_WINDOW"),
                        "recency_strength": CONFIG.get("RECENCY_STRENGTH"),
                    }
                })
            except Exception as e:
                results.append({"Aspect": aspect, "Sentiment": "Error", "error": str(e)})

        return JSONResponse(content={"result": results})
    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        return JSONResponse(content={"error": "Analysis failed", "details": error_msg}, status_code=500)


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)