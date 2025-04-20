import os
import requests
import logging
from typing import Optional, List, Dict, Any, Union
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import uvicorn
from travel_recommender import TravelRecommender

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize the travel recommender
recommender = TravelRecommender(data_dir="data", model_dir="models")

# Define response models
class HealthResponse(BaseModel):
    status: str
    model_trained: bool

class TrainRequest(BaseModel):
    retrain: bool = Field(default=False, description="Whether to force retraining even if a model exists")
    users_path: Optional[str] = Field(default=None, description="Path to the Users.xlsx file")
    places_path: Optional[str] = Field(default=None, description="Path to the Places.xlsx file")

class TrainResponse(BaseModel):
    success: bool
    message: str

class RecommendationResponse(BaseModel):
    success: bool
    user_id: str
    recommended_places: List[str]
    similar_users: List
    similarity_scores: List
    message: Optional[str] = None

class ErrorResponse(BaseModel):
    success: bool
    message: str
    recommendations: List = []

# Define lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load or train the model
    try:
        logger.info("Loading recommendation model...")
        if recommender.load_model():  # This should return True only on successful load
            logger.info("Model loaded successfully")
        else:
            logger.warning("Model loading returned False, will attempt to train a new model")
            if recommender.train():
                logger.info("New model trained successfully")
            else:
                logger.error("Failed to train new model")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        # Try to train a new model if loading fails
        try:
            logger.info("Training new model after load failure...")
            if recommender.train():
                logger.info("New model trained successfully")
            else:
                logger.error("Failed to train new model")
        except Exception as train_e:
            logger.error(f"Failed to train model: {str(train_e)}")
    
    yield  # Server is running and processing requests here
    
    # Shutdown: Perform any cleanup if needed
    logger.info("Shutting down API")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Travel Recommendation API",
    description="API for generating travel recommendations based on user preferences",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Health check endpoint
@app.get(
    "/health", 
    response_model=HealthResponse, 
    tags=["System"],
    summary="Check API health",
    description="Returns the status of the API and whether the recommendation model is trained"
)
async def health_check():
    return {
        "status": "ok",
        "model_trained": recommender.is_trained
    }

# Train model endpoint
@app.post(
    "/api/train", 
    response_model=TrainResponse, 
    tags=["Model Management"],
    summary="Train or retrain the recommendation model",
    description="Trains a new recommendation model or retrains an existing one"
)
async def train_model(request: TrainRequest):
    try:
        logger.info(f"Received request to train model. Retrain: {request.retrain}")
        
        # Train the model
        success = recommender.train(
            users_path=request.users_path, 
            places_path=request.places_path, 
            retrain=request.retrain
        )
        
        if success:
            logger.info("Model training successful")
            return {
                "success": True,
                "message": "Model trained successfully"
            }
        else:
            logger.error("Model training failed")
            raise HTTPException(
                status_code=500,
                detail="Model training failed"
            )
    
    except Exception as e:
        logger.error(f"Error training model: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error training model: {str(e)}"
        )

# Dependency for fetching preferences
async def get_user_preferences(user_id: str):
    try:
        preferences_url = f"https://travelguide.runasp.net/api/Preferences/{user_id}"
        logger.info(f"Fetching user preferences from: {preferences_url}")
        
        preferences_response = requests.get(preferences_url)
        preferences_response.raise_for_status()
        return preferences_response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching preferences: {str(e)}")
        raise HTTPException(
            status_code=502,
            detail=f"Error fetching user preferences: {str(e)}"
        )

# Dependency for fetching trips
async def get_user_trips(user_id: str):
    try:
        trips_url = f"https://travelguide.runasp.net/api/TripPlan/user/{user_id}"
        logger.info(f"Fetching user trips from: {trips_url}")
        
        trips_response = requests.get(trips_url)
        trips_response.raise_for_status()
        return trips_response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching trips: {str(e)}")
        raise HTTPException(
            status_code=502,
            detail=f"Error fetching user trips: {str(e)}"
        )

# Recommendations endpoint
@app.get(
    "/api/recommendations/{user_id}", 
    response_model=RecommendationResponse,
    responses={
        404: {"model": ErrorResponse, "description": "No recommendations generated"},
        502: {"model": ErrorResponse, "description": "Error fetching external data"},
        500: {"model": ErrorResponse, "description": "Server error"}
    },
    tags=["Recommendations"],
    summary="Get travel recommendations for a user",
    description="Fetches user preferences and past trips, then generates travel recommendations"
)
async def get_recommendations(
    user_id: str, 
    top_n: int = Query(5, description="Number of recommendations to return", ge=1, le=20),
    preferences: Dict = Depends(get_user_preferences),
    trips: Dict = Depends(get_user_trips)
):
    try:
        logger.info(f"Processing recommendation request for user: {user_id}")
        
        # Extract past cities from trips (if any)
        past_cities = []
        if "$values" in trips and trips["$values"]:
            past_cities = [trip.get("city", "") for trip in trips["$values"] if trip.get("city")]
            logger.info(f"User has {len(past_cities)} past trip cities: {past_cities}")
        else:
            logger.info(f"No past trips found for user {user_id}")
        
        # Prepare user data for the recommender
        user_data = {
            'Preferred Places': preferences.get("preferredPlaces", "") + (", " + ", ".join(past_cities) if past_cities else ""),
            'Travel Tags': preferences.get("travelTags", ""),
            'Age': preferences.get("age", 0),
            'Marital status': preferences.get("maritalStatus", ""),
            'Children': preferences.get("children", ""),
            'Gender': preferences.get("gender", "")
        }
        
        logger.info(f"Prepared user data for recommendation: {user_data}")
        
        # Get recommendations
        recommendations = recommender.recommend(user_data, top_n=top_n)
        
        if not recommendations:
            logger.warning(f"No recommendations generated for user {user_id}")
            raise HTTPException(
                status_code=404,
                detail="Could not generate recommendations. Please ensure the model is trained."
            )
        
        # Return recommendations to the frontend
        response = {
            "success": True,
            "user_id": user_id,
            "recommended_places": recommendations['recommendations'],
            "similar_users": recommendations['similar_users'],
            "similarity_scores": recommendations['similarity_scores']
        }
        
        logger.info(f"Generated recommendations for user {user_id}: {recommendations['recommendations']}")
        return response
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error generating recommendations: {str(e)}"
        )

# Add a simple root endpoint
@app.get("/", tags=["System"])
async def root():
    return {
        "message": "Travel Recommendation API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    # Get port from environment variable or use custom default (5050)
    port = int(os.environ.get('PORT', 5050))
    
    # Start the FastAPI app with uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)