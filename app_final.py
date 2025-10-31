import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['USE_TF'] = '0'
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import threading
import traceback
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import difflib
import math
import re
from contextlib import asynccontextmanager

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    def preload():
        try:
            load_absa_model()
        except Exception as e:
            print(f"Preload error: {e}")
    thread = threading.Thread(target=preload, daemon=True)
    thread.start()
    yield
    # Shutdown
    pass

app = FastAPI(title="Advanced ABSA System", lifespan=lifespan)

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
    "loading": False,
    "model_name": None
}

class AnalysisRequest(BaseModel):
    review: str
    aspects: Optional[List[str]] = None
    category: Optional[str] = "general"

# ============================================================================
# PRODUCT-SPECIFIC CONFIGURATIONS FOR 14 ELECTRONIC CATEGORIES
# ============================================================================

PRODUCT_CONFIGS = {
    "smartphones": {
        "NEGATION_WINDOW": 6,
        "FUZZY_THRESHOLD": 0.60,
        "EXACT_MATCH_BOOST": 1.6,
        "NEUTRAL_THRESHOLD": 0.12,
        "CONFIDENCE_THRESHOLD": 0.45,
        "RECENCY_WEIGHT": True,
        "RECENCY_STRENGTH": 0.3,
        "INTENSIFIERS_BOOST": 1.2,
        "common_aspects": ["battery", "camera", "screen", "performance", "price", "design", "storage"]
    },
    "laptops": {
        "NEGATION_WINDOW": 7,
        "FUZZY_THRESHOLD": 0.65,
        "EXACT_MATCH_BOOST": 1.7,
        "NEUTRAL_THRESHOLD": 0.15,
        "CONFIDENCE_THRESHOLD": 0.50,
        "RECENCY_WEIGHT": True,
        "RECENCY_STRENGTH": 0.25,
        "INTENSIFIERS_BOOST": 1.15,
        "common_aspects": ["battery life", "keyboard", "display", "performance", "build quality", "portability", "price"]
    },
    "tablets": {
        "NEGATION_WINDOW": 5,
        "FUZZY_THRESHOLD": 0.62,
        "EXACT_MATCH_BOOST": 1.5,
        "NEUTRAL_THRESHOLD": 0.13,
        "CONFIDENCE_THRESHOLD": 0.48,
        "RECENCY_WEIGHT": True,
        "RECENCY_STRENGTH": 0.28,
        "INTENSIFIERS_BOOST": 1.18,
        "common_aspects": ["screen", "battery", "performance", "price", "stylus", "apps", "portability"]
    },
    "smartwatches": {
        "NEGATION_WINDOW": 5,
        "FUZZY_THRESHOLD": 0.58,
        "EXACT_MATCH_BOOST": 1.4,
        "NEUTRAL_THRESHOLD": 0.14,
        "CONFIDENCE_THRESHOLD": 0.46,
        "RECENCY_WEIGHT": True,
        "RECENCY_STRENGTH": 0.32,
        "INTENSIFIERS_BOOST": 1.25,
        "common_aspects": ["battery life", "display", "fitness tracking", "notifications", "design", "price", "comfort"]
    },
    "headphones": {
        "NEGATION_WINDOW": 6,
        "FUZZY_THRESHOLD": 0.63,
        "EXACT_MATCH_BOOST": 1.55,
        "NEUTRAL_THRESHOLD": 0.12,
        "CONFIDENCE_THRESHOLD": 0.47,
        "RECENCY_WEIGHT": False,
        "RECENCY_STRENGTH": 0.0,
        "INTENSIFIERS_BOOST": 1.3,
        "common_aspects": ["sound quality", "noise cancellation", "comfort", "battery", "price", "build quality", "microphone"]
    },
    "cameras": {
        "NEGATION_WINDOW": 7,
        "FUZZY_THRESHOLD": 0.68,
        "EXACT_MATCH_BOOST": 1.8,
        "NEUTRAL_THRESHOLD": 0.16,
        "CONFIDENCE_THRESHOLD": 0.52,
        "RECENCY_WEIGHT": True,
        "RECENCY_STRENGTH": 0.22,
        "INTENSIFIERS_BOOST": 1.1,
        "common_aspects": ["image quality", "video quality", "autofocus", "battery", "build quality", "price", "features"]
    },
    "televisions": {
        "NEGATION_WINDOW": 6,
        "FUZZY_THRESHOLD": 0.64,
        "EXACT_MATCH_BOOST": 1.6,
        "NEUTRAL_THRESHOLD": 0.14,
        "CONFIDENCE_THRESHOLD": 0.49,
        "RECENCY_WEIGHT": True,
        "RECENCY_STRENGTH": 0.27,
        "INTENSIFIERS_BOOST": 1.2,
        "common_aspects": ["picture quality", "sound", "smart features", "price", "design", "size", "remote"]
    },
    "gaming_consoles": {
        "NEGATION_WINDOW": 6,
        "FUZZY_THRESHOLD": 0.62,
        "EXACT_MATCH_BOOST": 1.5,
        "NEUTRAL_THRESHOLD": 0.13,
        "CONFIDENCE_THRESHOLD": 0.47,
        "RECENCY_WEIGHT": True,
        "RECENCY_STRENGTH": 0.3,
        "INTENSIFIERS_BOOST": 1.25,
        "common_aspects": ["performance", "graphics", "games library", "price", "controller", "noise", "storage"]
    },
    "smart_speakers": {
        "NEGATION_WINDOW": 5,
        "FUZZY_THRESHOLD": 0.60,
        "EXACT_MATCH_BOOST": 1.45,
        "NEUTRAL_THRESHOLD": 0.13,
        "CONFIDENCE_THRESHOLD": 0.45,
        "RECENCY_WEIGHT": False,
        "RECENCY_STRENGTH": 0.0,
        "INTENSIFIERS_BOOST": 1.22,
        "common_aspects": ["sound quality", "voice assistant", "connectivity", "price", "design", "setup", "features"]
    },
    "routers": {
        "NEGATION_WINDOW": 7,
        "FUZZY_THRESHOLD": 0.66,
        "EXACT_MATCH_BOOST": 1.65,
        "NEUTRAL_THRESHOLD": 0.15,
        "CONFIDENCE_THRESHOLD": 0.50,
        "RECENCY_WEIGHT": True,
        "RECENCY_STRENGTH": 0.25,
        "INTENSIFIERS_BOOST": 1.15,
        "common_aspects": ["speed", "range", "reliability", "setup", "price", "features", "parental controls"]
    },
    "printers": {
        "NEGATION_WINDOW": 6,
        "FUZZY_THRESHOLD": 0.64,
        "EXACT_MATCH_BOOST": 1.6,
        "NEUTRAL_THRESHOLD": 0.14,
        "CONFIDENCE_THRESHOLD": 0.48,
        "RECENCY_WEIGHT": True,
        "RECENCY_STRENGTH": 0.28,
        "INTENSIFIERS_BOOST": 1.18,
        "common_aspects": ["print quality", "speed", "setup", "ink cost", "reliability", "price", "wireless"]
    },
    "monitors": {
        "NEGATION_WINDOW": 6,
        "FUZZY_THRESHOLD": 0.65,
        "EXACT_MATCH_BOOST": 1.6,
        "NEUTRAL_THRESHOLD": 0.15,
        "CONFIDENCE_THRESHOLD": 0.50,
        "RECENCY_WEIGHT": True,
        "RECENCY_STRENGTH": 0.26,
        "INTENSIFIERS_BOOST": 1.17,
        "common_aspects": ["picture quality", "color accuracy", "response time", "price", "build quality", "ports", "stand"]
    },
    "keyboards": {
        "NEGATION_WINDOW": 5,
        "FUZZY_THRESHOLD": 0.61,
        "EXACT_MATCH_BOOST": 1.5,
        "NEUTRAL_THRESHOLD": 0.12,
        "CONFIDENCE_THRESHOLD": 0.46,
        "RECENCY_WEIGHT": False,
        "RECENCY_STRENGTH": 0.0,
        "INTENSIFIERS_BOOST": 1.23,
        "common_aspects": ["typing feel", "build quality", "switches", "keycaps", "price", "layout", "lighting"]
    },
    "mice": {
        "NEGATION_WINDOW": 5,
        "FUZZY_THRESHOLD": 0.60,
        "EXACT_MATCH_BOOST": 1.45,
        "NEUTRAL_THRESHOLD": 0.12,
        "CONFIDENCE_THRESHOLD": 0.45,
        "RECENCY_WEIGHT": False,
        "RECENCY_STRENGTH": 0.0,
        "INTENSIFIERS_BOOST": 1.24,
        "common_aspects": ["sensor", "buttons", "comfort", "build quality", "price", "weight", "battery"]
    },
    "general": {
        "NEGATION_WINDOW": 6,
        "FUZZY_THRESHOLD": 0.63,
        "EXACT_MATCH_BOOST": 1.5,
        "NEUTRAL_THRESHOLD": 0.14,
        "CONFIDENCE_THRESHOLD": 0.48,
        "RECENCY_WEIGHT": True,
        "RECENCY_STRENGTH": 0.25,
        "INTENSIFIERS_BOOST": 1.2,
        "common_aspects": ["quality", "price", "features", "performance", "design", "value", "reliability"]
    }
}

# Base intensifiers
INTENSIFIERS = {
    "extremely": 1.6, "very": 1.35, "really": 1.3, "quite": 1.2, "pretty": 1.15,
    "absolutely": 1.55, "definitely": 1.35, "completely": 1.45, "totally": 1.4,
    "incredibly": 1.5, "highly": 1.35, "amazingly": 1.5, "exceptionally": 1.55,
    "slightly": 0.75, "somewhat": 0.82, "fairly": 0.88, "moderately": 0.85,
    "kind of": 0.78, "sort of": 0.78, "a bit": 0.82, "little": 0.8,
    "barely": 0.65, "hardly": 0.65, "scarcely": 0.65, "minimally": 0.7
}

NEGATIONS = {"not", "no", "n't", "never", "none", "nobody", "nothing", "nowhere",
             "hardly", "scarcely", "barely", "seldom", "rarely", "without",
             "lacks", "lacking", "missing", "fails", "failed", "doesn't", "don't",
             "didn't", "won't", "wouldn't", "shouldn't", "can't", "cannot"}

def get_config(product_category: str) -> Dict:
    """Get configuration for specific product category"""
    return PRODUCT_CONFIGS.get(product_category.lower().replace(" ", "_"), PRODUCT_CONFIGS["general"])

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    result = []
    for sent in sentences:
        if ';' in sent or ' - ' in sent or ' – ' in sent:
            result.extend([s.strip() for s in re.split(r'[;]|(?<=\S)\s+[-–]\s+', sent) if s.strip()])
        else:
            result.append(sent.strip())
    return [s for s in result if len(s) > 3]

def fuzzy_match_aspect(aspect: str, sentence: str) -> Tuple[float, bool]:
    """Return (score, exact_match) for aspect in sentence"""
    aspect_lower = aspect.lower()
    sentence_lower = sentence.lower()
    
    # Exact match
    if aspect_lower in sentence_lower:
        return 1.0, True
    
    # Clean for word matching
    clean_aspect = re.sub(r'[^\w\s]', '', aspect_lower).strip()
    clean_sentence = re.sub(r'[^\w\s]', '', sentence_lower).strip()
    
    if clean_aspect in clean_sentence:
        return 0.95, True
    
    # Multi-word matching
    aspect_words = clean_aspect.split()
    sentence_words = clean_sentence.split()
    
    if len(aspect_words) > 1 and all(aw in sentence_words for aw in aspect_words):
        return 0.88, False
    
    # Fuzzy scoring
    best_score = 0.0
    for i in range(len(sentence_words)):
        for j in range(i + 1, min(i + len(aspect_words) + 3, len(sentence_words) + 1)):
            phrase = ' '.join(sentence_words[i:j])
            score = difflib.SequenceMatcher(None, clean_aspect, phrase).ratio()
            best_score = max(best_score, score)
    
    # Single word fuzzy match
    if len(aspect_words) == 1:
        for word in sentence_words:
            score = difflib.SequenceMatcher(None, clean_aspect, word).ratio()
            best_score = max(best_score, score)
    
    return best_score, False

def has_negation_near(aspect: str, sentence: str, window: int = 6) -> bool:
    """Check for negation near aspect"""
    tokens = re.findall(r'\b\w+(?:\'[a-z]+)?\b', sentence.lower())
    aspect_tokens = re.findall(r'\b\w+\b', aspect.lower())
    
    if not aspect_tokens:
        return False
    
    for i in range(len(tokens)):
        match = True
        for j, asp_tok in enumerate(aspect_tokens):
            if i + j >= len(tokens) or not tokens[i + j].startswith(asp_tok[:min(3, len(asp_tok))]):
                match = False
                break
        
        if match:
            left = max(0, i - window)
            right = min(len(tokens), i + len(aspect_tokens) + window)
            for k in range(left, right):
                if tokens[k] in NEGATIONS or tokens[k].endswith("n't"):
                    return True
    return False

def get_intensity_multiplier(sentence: str, boost: float = 1.0) -> float:
    """Get intensity multiplier from sentence"""
    sentence_lower = sentence.lower()
    multiplier = 1.0
    
    for word, factor in INTENSIFIERS.items():
        if word in sentence_lower:
            multiplier *= (1 + (factor - 1) * boost)
    
    return max(0.5, min(multiplier, 2.0))

# ============================================================================
# MODEL FUNCTIONS
# ============================================================================

def map_label(label: str) -> str:
    """Map model labels to standard sentiment"""
    label_lower = label.lower()
    
    # Handle LABEL_0/LABEL_1 style
    if "label_0" in label_lower or label == "0":
        return "Negative"
    if "label_1" in label_lower or label == "1":
        return "Positive"
    
    # Handle text labels
    if "pos" in label_lower or label_lower == "positive":
        return "Positive"
    if "neg" in label_lower or label_lower == "negative":
        return "Negative"
    
    # Handle star ratings
    if "star" in label_lower:
        stars = int(label_lower.split()[0])
        if stars <= 2:
            return "Negative"
        elif stars >= 4:
            return "Positive"
        else:
            return "Neutral"
    
    return "Neutral"

def parse_model_output(raw: Any) -> Dict[str, float]:
    """Parse model output to probabilities"""
    if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], list):
        raw = raw[0]
    
    probs = {"Positive": 0.0, "Neutral": 0.0, "Negative": 0.0}
    
    if isinstance(raw, list):
        for item in raw:
            label = map_label(str(item.get("label", "")))
            score = float(item.get("score", 0.0))
            probs[label] += score
    
    # Normalize
    total = sum(probs.values()) or 1.0
    return {k: v / total for k, v in probs.items()}

def decide_sentiment(probs: Dict[str, float], threshold: float = 0.14) -> str:
    """Decide final sentiment"""
    pos = probs.get("Positive", 0.0)
    neg = probs.get("Negative", 0.0)
    neu = probs.get("Neutral", 0.0)
    
    if abs(pos - neg) < threshold:
        return "Neutral"
    
    max_prob = max(probs.values())
    if max_prob < 0.45:
        return "Neutral"
    
    if pos > neg and pos > neu:
        return "Positive"
    elif neg > pos and neg > neu:
        return "Negative"
    return "Neutral"

def load_absa_model():
    """Load sentiment model with fallbacks"""
    if MODEL["loaded"]:
        return True
    
    if MODEL["loading"]:
        return False
    
    MODEL["loading"] = True
    
    try:
        from transformers import pipeline
        
        models_to_try = [
            "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "distilbert-base-uncased-finetuned-sst-2-english",
            "cardiffnlp/twitter-roberta-base-sentiment",
            "nlptown/bert-base-multilingual-uncased-sentiment",
        ]
        
        classifier = None
        model_used = None
        
        for model_name in models_to_try:
            try:
                print(f"Attempting to load: {model_name}...")
                classifier = pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    device=-1,
                    return_all_scores=True,
                )
                model_used = model_name
                print(f"✓ Loaded: {model_name}")
                break
            except Exception as e:
                print(f"✗ Failed {model_name}: {str(e)[:80]}")
                continue
        
        if classifier is None:
            raise ValueError("No model could be loaded")
        
        # Test
        test_result = classifier("The product is good")
        if not test_result:
            raise ValueError("Model test failed")
        
        MODEL["classifier"] = classifier
        MODEL["model_name"] = model_used
        MODEL["loaded"] = True
        MODEL["loading"] = False
        print(f"✓ Model ready: {model_used}")
        return True
        
    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        MODEL["load_error"] = error_msg
        MODEL["loading"] = False
        print(f"✗ Load error: {error_msg}")
        return False

# ============================================================================
# CORE ANALYSIS ENGINE
# ============================================================================

def analyze_aspect_advanced(review: str, aspect: str, product_category: str = "general") -> Dict[str, Any]:
    """Advanced aspect-based sentiment analysis"""
    try:
        config = get_config(product_category)
        
        review = clean_text(review)
        aspect = clean_text(aspect)
        
        sentences = split_into_sentences(review)
        if not sentences:
            sentences = [review]
        
        # Find relevant sentences
        relevant = []
        for idx, sent in enumerate(sentences):
            fuzzy_score, exact = fuzzy_match_aspect(aspect, sent)
            if fuzzy_score >= config["FUZZY_THRESHOLD"]:
                weight = fuzzy_score
                if exact:
                    weight *= config["EXACT_MATCH_BOOST"]
                relevant.append((idx, sent, weight, fuzzy_score, exact))
        
        # Fallback with lower threshold
        if not relevant:
            for idx, sent in enumerate(sentences):
                fuzzy_score, exact = fuzzy_match_aspect(aspect, sent)
                if fuzzy_score >= 0.35:
                    relevant.append((idx, sent, fuzzy_score * 0.6, fuzzy_score, exact))
        
        # Last resort
        if not relevant:
            fuzzy_score, exact = fuzzy_match_aspect(aspect, review)
            relevant = [(0, review, max(0.25, fuzzy_score * 0.5), fuzzy_score, exact)]
        
        # Analyze each relevant sentence
        analyses = []
        for seq_num, (idx, sentence, base_weight, fuzzy_score, exact) in enumerate(relevant):
            # Build context
            context_parts = []
            if idx > 0:
                context_parts.append(sentences[idx - 1])
            context_parts.append(sentence)
            if idx < len(sentences) - 1:
                context_parts.append(sentences[idx + 1])
            context = ' '.join(context_parts)
            
            # Get prediction
            try:
                input_text = f"{context} [SEP] Aspect: {aspect}"
                predictions = MODEL["classifier"](input_text, truncation=True, max_length=512)
            except:
                predictions = MODEL["classifier"](context, truncation=True, max_length=512)
            
            probs = parse_model_output(predictions)
            
            # Apply heuristics
            intensity = get_intensity_multiplier(sentence, config["INTENSIFIERS_BOOST"])
            has_neg = has_negation_near(aspect, sentence, config["NEGATION_WINDOW"])
            
            # Negation adjustment
            if has_neg:
                adjusted = {
                    "Positive": probs["Negative"] * 0.85,
                    "Negative": probs["Positive"] * 0.85,
                    "Neutral": probs["Neutral"] * 1.3
                }
                total = sum(adjusted.values())
                probs = {k: v / total for k, v in adjusted.items()}
            
            # Intensity adjustment
            if intensity != 1.0:
                dominant = max(probs, key=probs.get)
                if dominant in ["Positive", "Negative"]:
                    probs[dominant] = min(0.98, probs[dominant] * intensity)
                    total = sum(probs.values())
                    probs = {k: v / total for k, v in probs.items()}
            
            # Confidence weighting
            confidence = max(probs.values())
            conf_weight = math.exp((confidence - 0.5) * 1.8)
            final_weight = base_weight * conf_weight
            
            # Recency weighting
            if config["RECENCY_WEIGHT"] and len(relevant) > 1:
                recency = 1.0 + (seq_num / (len(relevant) - 1)) * config["RECENCY_STRENGTH"]
                final_weight *= recency
            
            analyses.append({
                "sentence": sentence,
                "probs": probs,
                "weight": final_weight,
                "confidence": confidence,
                "has_negation": has_neg,
                "intensity": intensity,
                "fuzzy_score": fuzzy_score,
                "exact_match": exact
            })
        
        # Aggregate
        total_weight = sum(a["weight"] for a in analyses) or 1.0
        agg_probs = {"Positive": 0.0, "Neutral": 0.0, "Negative": 0.0}
        
        for analysis in analyses:
            for sent, prob in analysis["probs"].items():
                agg_probs[sent] += prob * analysis["weight"]
        
        for sent in agg_probs:
            agg_probs[sent] /= total_weight
        
        # Final sentiment
        sentiment = decide_sentiment(agg_probs, config["NEUTRAL_THRESHOLD"])
        
        # Score (1-10 scale)
        score = 5.5 + 4.5 * (agg_probs["Positive"] - agg_probs["Negative"])
        score = max(1.0, min(10.0, score))
        
        # Confidence
        confidence = max(agg_probs.values()) * 100
        
        return {
            "Aspect": aspect,
            "Sentiment": sentiment,
            "Probabilities": {
                "Positive": round(agg_probs["Positive"], 4),
                "Neutral": round(agg_probs["Neutral"], 4),
                "Negative": round(agg_probs["Negative"], 4),
            },
            "Score (1-10)": round(score, 2),
            "Confidence (%)": round(confidence, 1),
            "Details": {
                "sentences_analyzed": len(analyses),
                "exact_matches": sum(1 for a in analyses if a["exact_match"]),
                "negations_found": sum(1 for a in analyses if a["has_negation"]),
                "avg_intensity": round(sum(a["intensity"] for a in analyses) / len(analyses), 2) if analyses else 1.0,
                "product_category": product_category,
                "config_used": {
                    "fuzzy_threshold": config["FUZZY_THRESHOLD"],
                    "negation_window": config["NEGATION_WINDOW"]
                }
            }
        }
        
    except Exception as e:
        return {
            "Aspect": aspect,
            "Sentiment": "Error",
            "Score (1-10)": 5.0,
            "Confidence (%)": 0.0,
            "Probabilities": {"Positive": 0.0, "Neutral": 0.0, "Negative": 0.0},
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def home():
    """Serve main page"""
    index_path = static_dir / "index_advanced.html"
    if index_path.exists():
        return FileResponse(index_path)
    # Fallback to basic index
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return JSONResponse({"error": "Frontend not found"}, status_code=404)

@app.get("/status")
async def status():
    """Check status"""
    return JSONResponse({
        "status": "ready" if MODEL["loaded"] else "loading",
        "model_loaded": MODEL["loaded"],
        "loading": MODEL["loading"],
        "load_error": MODEL["load_error"],
        "model": MODEL["model_name"] or "Loading...",
        "categories_supported": len(PRODUCT_CONFIGS)
    })

@app.get("/categories")
async def get_categories():
    """Get product categories"""
    categories = []
    for key, config in PRODUCT_CONFIGS.items():
        categories.append({
            "id": key,
            "name": key.replace("_", " ").title(),
            "common_aspects": config.get("common_aspects", [])
        })
    return JSONResponse({"categories": categories})

@app.post("/analyze")
async def analyze_absa(request: AnalysisRequest):
    """Analyze review"""
    try:
        review = request.review.strip()
        aspects_input = request.aspects
        product_category = request.category or "general"
        
        if not review:
            return JSONResponse(
                {"error": "Review is required"},
                status_code=400
            )
        
        if not MODEL["loaded"]:
            if not load_absa_model():
                return JSONResponse(
                    {"error": "Model not loaded", "details": MODEL["load_error"]},
                    status_code=500
                )
        
        # Handle aspects input (can be None, list, or string)
        aspects_list = []
        if aspects_input:
            if isinstance(aspects_input, list):
                aspects_list = [a.strip() for a in aspects_input if a and a.strip()]
            elif isinstance(aspects_input, str):
                aspects_list = [a.strip() for a in aspects_input.split(",") if a.strip()]
        
        # If no aspects provided, use common aspects for the category
        if not aspects_list and product_category in PRODUCT_CONFIGS:
            aspects_list = PRODUCT_CONFIGS[product_category].get("common_aspects", [])[:5]
        
        if not aspects_list:
            return JSONResponse(
                {"error": "No valid aspects to analyze"},
                status_code=400
            )
        
        results = []
        for aspect in aspects_list:
            result = analyze_aspect_advanced(review, aspect, product_category)
            results.append(result)
        
        return JSONResponse(results)
        
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            {"error": "Analysis failed", "details": str(e)},
            status_code=500
        )

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 Advanced ABSA System - Production Ready")
    print("="*70)
    print(f"📊 Supported Categories: {len(PRODUCT_CONFIGS)}")
    print(f"🎯 Models: Multi-fallback system")
    print(f"🌐 Server: http://127.0.0.1:8000")
    print("="*70 + "\n")
    
    uvicorn.run("app_final:app", host="127.0.0.1", port=8000, reload=True)
