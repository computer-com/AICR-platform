"""
AICR Platform - FastAPI Backend
Provides REST API for crisis report submission and ML predictions
"""

import os
import sys
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import json

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Try to import ML predictor
try:
    from src.ml_models.predict import CrisisPredictor
    ML_AVAILABLE = True
except ImportError:
    print("[WARNING] Could not import CrisisPredictor from src.ml_models.predict")
    print("[INFO] Attempting alternative import...")
    try:
        # Add src directory to path
        src_dir = os.path.join(project_root, 'src')
        sys.path.insert(0, src_dir)
        from ml_models.predict import CrisisPredictor
        ML_AVAILABLE = True
    except ImportError as e:
        print(f"[ERROR] Could not import ML predictor: {e}")
        print("[INFO] API will run in demo mode without ML predictions")
        ML_AVAILABLE = False
        CrisisPredictor = None

# Initialize FastAPI app
app = FastAPI(
    title="AICR Crisis Response API",
    description="AI-Augmented Crisis Response Platform API",
    version="1.0.0"
)

# CORS middleware - allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ML model
predictor = None
if ML_AVAILABLE and CrisisPredictor:
    try:
        predictor = CrisisPredictor(model_type='multimodal')
        print("[OK] Multimodal model loaded successfully")
    except Exception as e:
        print(f"[INFO] Could not load multimodal model: {e}")
        try:
            predictor = CrisisPredictor(model_type='text')
            print("[OK] Text model loaded successfully")
        except Exception as e2:
            print(f"[WARNING] Could not load text model: {e2}")
            print("[INFO] Running in demo mode - predictions will be simulated")
else:
    print("[INFO] Running in demo mode - ML models not available")

# In-memory storage for demo (in production, use database)
incidents_db = []
incident_id_counter = 1

# ---- Pydantic Models ----

class CrisisReport(BaseModel):
    """Input model for citizen crisis reports"""
    description: str = Field(..., min_length=10, description="Description of the crisis")
    latitude: float = Field(..., ge=-90, le=90, description="Latitude coordinate")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude coordinate")
    source: str = Field(default="citizen", description="Report source (citizen/social)")
    image_path: Optional[str] = Field(None, description="Optional image path")

class PredictionResponse(BaseModel):
    """Response model for ML predictions"""
    crisis_type: str
    confidence: float
    all_predictions: List[dict]

class IncidentResponse(BaseModel):
    """Response model for stored incidents"""
    id: int
    description: str
    latitude: float
    longitude: float
    crisis_type: str
    confidence: float
    timestamp: str
    source: str
    status: str
    image_path: Optional[str]

class HotspotResponse(BaseModel):
    """Response model for crisis hotspots"""
    latitude: float
    longitude: float
    crisis_type: str
    count: int
    avg_confidence: float

# ---- API Endpoints ----

@app.get("/")
async def root():
    """Root endpoint - API health check"""
    return {
        "message": "AICR Crisis Response API",
        "version": "1.0.0",
        "status": "operational",
        "model_loaded": predictor is not None,
        "endpoints": {
            "predict": "/predict",
            "submit_report": "/report",
            "get_incidents": "/incidents",
            "get_hotspots": "/hotspots",
            "stats": "/stats"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_status": "loaded" if predictor else "not_loaded",
        "incidents_count": len(incidents_db)
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_crisis(report: CrisisReport):
    """
    Predict crisis type from description and location
    
    Args:
        report: Crisis report with description and coordinates
        
    Returns:
        Prediction with crisis type, confidence, and all probabilities
    """
    if predictor is None:
        # Demo mode - use improved keyword-based prediction
        predicted_type, confidence, all_preds = keyword_based_prediction(report.description)
        
        return PredictionResponse(
            crisis_type=predicted_type,
            confidence=confidence,
            all_predictions=all_preds
        )
    
    try:
        result = predictor.predict_text(
            report.description,
            report.latitude,
            report.longitude
        )
        
        return PredictionResponse(
            crisis_type=result['predicted_class'],
            confidence=result['confidence'],
            all_predictions=result['all_predictions']
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/report", response_model=IncidentResponse)
async def submit_report(report: CrisisReport, background_tasks: BackgroundTasks):
    """
    Submit a new crisis report
    
    Args:
        report: Crisis report details
        background_tasks: FastAPI background tasks
        
    Returns:
        Created incident with prediction results
    """
    global incident_id_counter
    
    try:
        # Get prediction (using demo mode if model not loaded)
        pred_response = await predict_crisis(report)
        
        # Create incident record
        incident = {
            "id": incident_id_counter,
            "description": report.description,
            "latitude": report.latitude,
            "longitude": report.longitude,
            "crisis_type": pred_response.crisis_type,
            "confidence": pred_response.confidence,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "source": report.source,
            "status": "active",
            "image_path": report.image_path
        }
        
        incidents_db.append(incident)
        incident_id_counter += 1
        
        # Optional: log to file in background
        background_tasks.add_task(log_incident, incident)
        
        return IncidentResponse(**incident)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report submission failed: {str(e)}")

@app.get("/incidents", response_model=List[IncidentResponse])
async def get_incidents(
    limit: int = 100,
    crisis_type: Optional[str] = None,
    status: Optional[str] = None
):
    """
    Get all incidents with optional filtering
    
    Args:
        limit: Maximum number of incidents to return
        crisis_type: Filter by crisis type
        status: Filter by status (active/resolved)
        
    Returns:
        List of incidents
    """
    filtered = incidents_db
    
    # Apply filters
    if crisis_type:
        filtered = [i for i in filtered if i['crisis_type'] == crisis_type]
    if status:
        filtered = [i for i in filtered if i['status'] == status]
    
    # Return latest incidents first
    filtered = sorted(filtered, key=lambda x: x['timestamp'], reverse=True)
    
    return [IncidentResponse(**inc) for inc in filtered[:limit]]

@app.get("/incidents/{incident_id}", response_model=IncidentResponse)
async def get_incident(incident_id: int):
    """
    Get a specific incident by ID
    
    Args:
        incident_id: Incident ID
        
    Returns:
        Incident details
    """
    incident = next((i for i in incidents_db if i['id'] == incident_id), None)
    
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    return IncidentResponse(**incident)

@app.put("/incidents/{incident_id}/status")
async def update_incident_status(incident_id: int, status: str):
    """
    Update incident status
    
    Args:
        incident_id: Incident ID
        status: New status (active/resolved/investigating)
        
    Returns:
        Updated incident
    """
    incident = next((i for i in incidents_db if i['id'] == incident_id), None)
    
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    if status not in ['active', 'resolved', 'investigating']:
        raise HTTPException(status_code=400, detail="Invalid status")
    
    incident['status'] = status
    
    return IncidentResponse(**incident)

@app.get("/hotspots", response_model=List[HotspotResponse])
async def get_hotspots(radius: float = 0.01):
    """
    Get crisis hotspots (areas with multiple incidents)
    
    Args:
        radius: Clustering radius in degrees (~1km = 0.01 degrees)
        
    Returns:
        List of hotspots with aggregated data
    """
    if not incidents_db:
        return []
    
    # Simple clustering by rounding coordinates
    from collections import defaultdict
    
    clusters = defaultdict(list)
    
    for incident in incidents_db:
        if incident['status'] == 'active':
            # Round coordinates to cluster nearby incidents
            lat_key = round(incident['latitude'] / radius) * radius
            lon_key = round(incident['longitude'] / radius) * radius
            key = (lat_key, lon_key, incident['crisis_type'])
            clusters[key].append(incident)
    
    # Build hotspot response
    hotspots = []
    for (lat, lon, crisis_type), incidents in clusters.items():
        if len(incidents) >= 2:  # Only include clusters with 2+ incidents
            avg_confidence = sum(i['confidence'] for i in incidents) / len(incidents)
            hotspots.append(HotspotResponse(
                latitude=lat,
                longitude=lon,
                crisis_type=crisis_type,
                count=len(incidents),
                avg_confidence=avg_confidence
            ))
    
    return hotspots

@app.get("/stats")
async def get_statistics():
    """
    Get platform statistics
    
    Returns:
        Statistics about incidents and predictions
    """
    if not incidents_db:
        return {
            "total_incidents": 0,
            "active_incidents": 0,
            "crisis_types": {},
            "avg_confidence": 0
        }
    
    # Calculate statistics
    from collections import Counter
    
    crisis_counts = Counter(i['crisis_type'] for i in incidents_db)
    active_count = sum(1 for i in incidents_db if i['status'] == 'active')
    avg_confidence = sum(i['confidence'] for i in incidents_db) / len(incidents_db)
    
    return {
        "total_incidents": len(incidents_db),
        "active_incidents": active_count,
        "resolved_incidents": len(incidents_db) - active_count,
        "crisis_types": dict(crisis_counts),
        "avg_confidence": round(avg_confidence, 3),
        "latest_incident": incidents_db[-1]['timestamp'] if incidents_db else None
    }

@app.delete("/incidents/clear")
async def clear_incidents():
    """Clear all incidents (for testing)"""
    global incidents_db, incident_id_counter
    incidents_db = []
    incident_id_counter = 1
    return {"message": "All incidents cleared"}

def keyword_based_prediction(description: str) -> tuple:
    """Improved keyword-based crisis classification (fallback when ML model unavailable)"""
    crisis_types = ['flood', 'fire', 'earthquake', 'gas_leak', 'storm', 'power_outage', 'landslide']
    
    description_lower = description.lower()
    
    # Weighted keyword matching - ORDER MATTERS (check specific phrases first)
    keywords = {
        'flood': {
            'primary': ['flood', 'flooding', 'flooded', 'inundation', 'submerged', 'waterlogged'],
            'secondary': ['rising water', 'overflow', 'river bank', 'dam breach', 'heavy rain', 'water level']
        },
        'fire': {
            'primary': ['fire', 'burning', 'flames', 'blaze', 'inferno', 'combustion'],
            'secondary': ['smoke', 'burnt', 'ash', 'heat', 'ignite', 'spreading']
        },
        'earthquake': {
            'primary': ['earthquake', 'quake', 'tremor', 'seismic'],
            'secondary': ['shaking', 'aftershock', 'ground shaking', 'magnitude', 'epicenter', 'building collapse']
        },
        'gas_leak': {
            'primary': ['gas leak', 'leaking gas', 'gas explosion'],
            'secondary': ['gas smell', 'odor', 'smell gas', 'pipeline', 'propane', 'natural gas', 'fumes']
        },
        'storm': {
            'primary': ['storm', 'hurricane', 'tornado', 'cyclone', 'typhoon'],
            'secondary': ['high wind', 'lightning', 'thunder', 'hail', 'severe weather', 'gust']
        },
        'power_outage': {
            'primary': ['power outage', 'blackout', 'power failure', 'no electricity'],
            'secondary': ['no power', 'electricity out', 'transformer', 'grid failure', 'lights out']
        },
        'landslide': {
            'primary': ['landslide', 'mudslide', 'rockslide', 'debris flow'],
            'secondary': ['slope failure', 'hillside collapse', 'erosion', 'avalanche', 'mud']
        }
    }
    
    # Score each crisis type with weighted scoring
    scores = {}
    matched_keywords = {}
    
    for crisis_type, word_dict in keywords.items():
        score = 0
        matches = []
        
        # Check primary keywords first (worth 10 points each)
        for word in word_dict['primary']:
            if word in description_lower:
                score += 10
                matches.append(word)
        
        # Check secondary keywords (worth 2 points each)
        for word in word_dict['secondary']:
            if word in description_lower:
                score += 2
                matches.append(word)
        
        if score > 0:
            scores[crisis_type] = score
            matched_keywords[crisis_type] = matches
    
    # Debug logging
    print(f"\n[DEBUG] Description: {description}")
    print(f"[DEBUG] Scores: {scores}")
    print(f"[DEBUG] Matched keywords: {matched_keywords}")
    
    # Determine predicted type
    if scores:
        predicted_type = max(scores, key=scores.get)
        max_score = scores[predicted_type]
        
        # Check if we have primary keyword match
        has_primary = any(word in description_lower for word in keywords[predicted_type]['primary'])
        
        # Calculate confidence based on score and primary match
        if has_primary and max_score >= 10:
            confidence = min(0.95, 0.80 + (max_score * 0.01))
        elif max_score >= 5:
            confidence = min(0.75, 0.60 + (max_score * 0.02))
        else:
            confidence = min(0.65, 0.50 + (max_score * 0.03))
    else:
        # No keywords matched - random guess with low confidence
        import random
        predicted_type = random.choice(crisis_types)
        confidence = random.uniform(0.30, 0.40)
        print(f"[DEBUG] No keywords matched! Random guess: {predicted_type}")
    
    # Generate all predictions
    all_preds = []
    total_score = sum(scores.values()) if scores else 1
    
    for ct in crisis_types:
        if ct == predicted_type:
            all_preds.append({'crisis_type': ct, 'confidence': confidence})
        else:
            ct_score = scores.get(ct, 0)
            if ct_score > 0:
                # Other types with matches get proportional confidence
                ct_confidence = min(0.40, (ct_score / (max_score + 1)) * confidence * 0.6)
            else:
                # No matches get very low confidence
                import random
                ct_confidence = random.uniform(0.01, 0.05)
            all_preds.append({'crisis_type': ct, 'confidence': ct_confidence})
    
    # Sort by confidence
    all_preds = sorted(all_preds, key=lambda x: x['confidence'], reverse=True)
    
    # Normalize confidences to sum to 1.0
    conf_sum = sum(p['confidence'] for p in all_preds)
    if conf_sum > 0:
        for pred in all_preds:
            pred['confidence'] = round(pred['confidence'] / conf_sum, 4)
    
    print(f"[DEBUG] Final prediction: {predicted_type} ({all_preds[0]['confidence']:.2%})")
    
    return predicted_type, all_preds[0]['confidence'], all_preds

# ---- Helper Functions ----

def log_incident(incident: dict):
    """Log incident to file (background task)"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "incidents.log")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(incident) + "\n")

# ---- Startup Event ----

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    print("="*60)
    print("AICR Crisis Response API Starting...")
    print("="*60)
    print(f"Model Status: {'Loaded' if predictor else 'Not Loaded'}")
    print(f"Endpoints: http://localhost:8000/docs")
    print("="*60)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)