# AI Real Estate Agent 

**Intelligent property price prediction & market insights for Ames, Iowa**

An LLM-powered real estate agent that combines natural language processing with machine learning to provide instant property valuations and market analysis.

---

## Features

### **Price Prediction**
- Describe a property in plain English
- AI extracts key features automatically
- ML model predicts market price instantly
- LLM provides market context & insights

### *Market Insights**
- Ask about neighborhoods, trends, value propositions
- Real data from 2,930+ historical sales
- Visualize price distributions by quality, location, features
- Compare neighborhoods and identify opportunities

### **Smart Feature Extraction**
- Two-stage LLM pipeline (Stage 1: Extract, Stage 2: Interpret)
- Handles vague descriptions gracefully
- Maps natural language to structured features
- Validates input against historical data

---

## Architecture

### Tech Stack
- **Backend**: FastAPI + Python
- **Frontend**: Vanilla JavaScript + Chart.js
- **ML Model**: Gradient Boosting Regressor (sklearn)
- **LLM**: OpenAI GPT-4o-mini
- **Data**: Ames Housing CSV (Iowa, 2006-2010)

### Request Flow
```
User Input
    ↓
[Intent Classifier] → Is this a prediction or insight?
    ↓
    ├→ PREDICTION:
    │   ├ Stage 1: Extract Features (LLM)
    │   ├ User Form: Fill missing fields
    │   ├ Stage 2: ML Prediction
    │   ├ Stage 3: Interpret (LLM + Market Context)
    │   └ Display: Price + Breakdown
    │
    └→ INSIGHT:
        ├ Real Training Stats
        ├ Topic Detection
        ├ Generate Analysis (LLM)
        └ Display: Findings + Charts
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- OpenAI API key (get free credits at openai.com)

### 1. Backend Setup

```bash
cd backend

# Install dependencies
pip install -r requirements.txt
# or with uv: uv sync

# Create .env with your keys
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=sk-...

# Start server
uv run uvicorn app.main:app --reload --port 8000
```

Server runs at: `http://localhost:8000`

Health check: `curl http://localhost:8000/health`

### 2. Frontend Setup

```bash
cd frontend

# Open in browser (or use live server)
open index.html
```

Frontend connects to: `http://127.0.0.1:8000`

---

## Try These Examples

### Price Predictions
- *"3-bedroom ranch with a 2-car garage in a nice neighborhood, built around 2000"*
- *"Luxury 4-bed house, excellent kitchen, large basement, NridgHt"*
- *"Small starter home, 2 bed 1 bath, affordable neighborhood, fixer-upper"*

### Market Questions
- *"What are the most expensive neighborhoods in Ames?"*
- *"How does kitchen quality affect home value?"*
- *"Which neighborhoods give the best value for money?"*

---

##  Data

**Ames Housing Dataset**
- **Size**: 2,930 residential sales
- **Time Period**: 2006-2010
- **Features**: 80 original → 10 selected
- **Neighborhoods**: 22 (premium to affordable)
- **Price Range**: $34k–$755k (median: $160k)

**Selected Features for Prediction**
1. Overall Quality (1-10 scale)
2. Living Area (sqft)
3. Basement Area (sqft)
4. Garage Capacity (cars)
5. Full Bathrooms
6. Year Built
7. Lot Area (sqft)
8. Neighborhood
9. Exterior Quality
10. Kitchen Quality

---

## Project Structure

```
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app entry point
│   │   ├── config.py            # Environment setup
│   │   ├── llm/
│   │   │   ├── intent_classifier.py        # Stage 0
│   │   │   ├── feature_extractor_stage1.py # Stage 1
│   │   │   └── price_interpreter_stage2.py # Stage 2
│   │   ├── ml/
│   │   │   └── price_predictor.py  # Model loading + inference
│   │   ├── routes/
│   │   │   ├── prediction_route.py # /classify, /extract, /predict
│   │   │   └── insights_route.py   # /insights, /stats
│   │   └── schemas/
│   │       ├── house_features_schema.py
│   │       └── api_response_schema.py
│   ├── saved_model/
│   │   ├── trained_house_price_model.pkl
│   │   └── training_statistics.json
│   └── pyproject.toml
├── frontend/
│   ├── index.html       # Single-page app with all UI/JS
│   ├── .env             # Backend URL config
│   └── vercel.json      # Deployment config
└── model_training_notebook/
    └── AI_REAL_ESTATE_AGENT.ipynb  # Model training (Jupyter)
```

---

## API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/api/v1/classify` | Classify user intent |
| `POST` | `/api/v1/extract` | Extract features from text |
| `POST` | `/api/v1/predict` | Predict price given features |
| `POST` | `/api/v1/insights` | Answer market questions |
| `GET` | `/api/v1/stats` | Fetch training statistics |
| `GET` | `/health` | Health check |

### Example: Price Prediction Flow

```bash
# 1. Classify intent
curl -X POST http://localhost:8000/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"user_query": "3-bed house with garage"}'

# Response: {"intent": "prediction", "confidence": 0.95, ...}

# 2. Extract features
curl -X POST http://localhost:8000/api/v1/extract \
  -H "Content-Type: application/json" \
  -d '{"user_query": "3-bed house with garage, built 2005, good condition"}'

# 3. Submit for prediction
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "overall_quality": 7,
    "gr_liv_area": 1500,
    "garage_cars": 2,
    ...
  }'

# Response: {"predicted_price": 185000, "interpretation": "...", ...}
```

---

##  How It Works

### Stage 0: Intent Classification
- Classifier LLM routes query → "prediction", "insights", or "other"
- Defends against prompt injection & off-topic queries
- Fails gracefully with friendly message

### Stage 1: Feature Extraction (LLM)
- Converts natural language → structured features
- Two prompt variants for A/B testing
- Assigns completeness score (0.0–1.0)
- Frontend shows form for missing fields

### Stage 2: ML Prediction (Model)
- Full sklearn pipeline (preprocessing + GB regressor)
- Imputes missing values
- Scales & encodes features
- Outputs: predicted price + confidence bounds

### Stage 3: LLM Interpretation
- Contextualizes prediction with market data
- Compares to neighborhood/quality average
- Identifies key price drivers
- Generates natural language insight

---

## Model Performance

**Validation Set:**
- RMSE: $27,500
- R² Score: 0.91
- (Based on Ames historical data)

**Key Predictors (by importance):**
1. Overall Quality (21%)
2. Living Area (15%)
3. Basement Area (9%)
4. Year Built (5%)
5. Lot Area (4%)

---

##  Assignment Notes

**What This Demonstrates:**
✅ LLM prompt chaining (3 stages)  
✅ ML model integration  
✅ Feature extraction & validation  
✅ API design & error handling  
✅ Frontend-backend communication  
✅ Real-world data workflow  

**Key Technologies:**
- FastAPI (async, type-safe REST API)
- Pydantic (data validation)
- OpenAI API (LLM integration)
- Scikit-learn (ML model)
- Vanilla JS (frontend)

---

##  Deployment

### Backend (Railway)

### Frontend (Vercel)
Push `frontend/` folder to Vercel—automatic deployment.

---

##  Support

**Issues?**
1. Ensure backend is running: `http://localhost:8000/health`
2. Check OPENAI_API_KEY is set in `.env`

---


**Built with ❤️ using LLM + ML** 
