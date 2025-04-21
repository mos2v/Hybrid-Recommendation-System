from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import os
import logging
import requests
from datetime import datetime, timedelta
import asyncio
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
    last_update: str

class TrainRequest(BaseModel):
    retrain: bool = Field(default=False, description="Whether to force retraining even if a model exists")
    users_path: Optional[str] = Field(default=None, description="Path to the Users.xlsx file")
    places_path: Optional[str] = Field(default=None, description="Path to the Places.xlsx file")

class TrainResponse(BaseModel):
    success: bool
    message: str

class RecommendationRequest(BaseModel):
    preferred_places: Optional[str] = Field(default="", description="Comma-separated list of preferred places")
    travel_tags: Optional[str] = Field(default="", description="Comma-separated list of travel tags")
    age: Optional[int] = Field(default=0, description="User age")
    marital_status: Optional[str] = Field(default="", description="User marital status")
    children: Optional[str] = Field(default="", description="User children status")
    gender: Optional[str] = Field(default="", description="User gender")

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

class UserUpdateRequest(BaseModel):
    trip_data: Dict = Field(..., description="Trip data including city name")

class UserUpdateResponse(BaseModel):
    success: bool
    message: str
    updated_at: str

class BatchUpdateRequest(BaseModel):
    user_list: List[str] = Field(..., description="List of user IDs to update")

class BatchUpdateResponse(BaseModel):
    success: bool
    updated_count: int
    message: str

class ModelMetadataResponse(BaseModel):
    last_global_update: str
    user_count: int
    updated_users_24h: int
    total_trips_processed: int

class TripPlan(BaseModel):
    id: int
    city: str
    image: str
    visitorType: str
    numDays: str
    budget: str
    userId: str
    tripDays: Optional[List[Dict]] = None

# Define lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Loading recommendation model...")
    if recommender.load_model():  # This should return True only on successful load
        logger.info("Model loaded successfully")
    else:
        logger.warning("Model loading returned False, will attempt to train a new model")
        if recommender.train():
            logger.info("New model trained successfully")
        else:
            logger.error("Failed to train new model")
    
    # Set up background tasks
    background_tasks = BackgroundTasks()
    background_tasks.add_task(update_active_users_periodically)
    logger.info("Background tasks scheduled")
    
    
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


async def update_active_users_periodically():
    """
    Periodically update users with recent activity
    """
    while True:
        try:
            logger.info("Starting periodic update of active users")
            
            # Get all users who have trips
            all_trips = await get_all_trips()
            
            # Get timestamp of last update
            last_update_str = recommender.recommender.get_last_global_update()
            if last_update_str:
                last_update = datetime.fromisoformat(last_update_str)
            else:
                last_update = datetime.utcnow() - timedelta(days=7)  # Default to 7 days ago
            
            # Track users who need updates
            users_to_update = []
            
            # Get unique user IDs from trips
            processed_users = set()
            for trip in all_trips:
                user_id = trip.get("userId")
                
                # Skip if user already processed or no ID
                if not user_id or user_id in processed_users:
                    continue
                    
                processed_users.add(user_id)
                
                # Check if user needs update - if they have trips but metadata timestamp is old
                if user_id in recommender.recommender.user_metadata:
                    user_last_update_str = recommender.recommender.user_metadata[user_id]
                    try:
                        user_last_update = datetime.fromisoformat(user_last_update_str)
                        if user_last_update < last_update:
                            users_to_update.append(user_id)
                    except (ValueError, TypeError):
                        # If timestamp is invalid, update the user
                        users_to_update.append(user_id)
                else:
                    # If user has no metadata entry, they need an update
                    users_to_update.append(user_id)
            
            # Update users
            updated_count = 0
            for user_id in users_to_update:
                try:
                    logger.info(f"Updating user {user_id}")
                    
                    # Get user preferences and trips
                    user_prefs = await get_user_preferences(user_id)
                    user_trips = await get_user_trips(user_id)
                    
                    # Prepare user data
                    user_data = prepare_user_data(user_prefs, user_trips)
                    
                    # Update user
                    if recommender.recommender.update_user(user_id, user_data, recommender.vectorizer):
                        updated_count += 1
                        logger.info(f"Successfully updated user {user_id}")
                except Exception as e:
                    logger.error(f"Error updating user {user_id}: {str(e)}")
            
            # Update global timestamp if any users were updated
            if updated_count > 0:
                recommender.recommender.update_global_timestamp()
                recommender.save_model()
                logger.info(f"Batch update completed: {updated_count} users updated")
            else:
                logger.info("No users needed updates")
                
        except Exception as e:
            logger.error(f"Error in batch update: {str(e)}")
        
        # Sleep for 6 hours before next update
        await asyncio.sleep(6 * 3600)  # 6 hours in seconds

async def get_all_trips():
    """
    Fetch all trips from the TripPlan API
    
    Returns:
    - List of trip objects
    """
    try:
        trips_url = "https://travelguide.runasp.net/api/TripPlan"
        logger.info(f"Fetching all trips from: {trips_url}")
        
        trips_response = requests.get(trips_url)
        trips_response.raise_for_status()
        trips_data = trips_response.json()
        
        # Extract values from the response
        if "$values" in trips_data:
            return trips_data["$values"]
        else:
            logger.warning("No $values field in trips response")
            return []
    except Exception as e:
        logger.error(f"Error fetching all trips: {str(e)}")
        return []

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

def prepare_user_data(preferences, trips):
    """
    Prepare user data from preferences and trips
    
    Parameters:
    - preferences: User preferences data
    - trips: User trips data
    
    Returns:
    - Dict of user data formatted for the recommender
    """
    # Extract past cities from trips (if any)
    past_cities = []
    
    if isinstance(trips, dict) and "$values" in trips and trips["$values"]:
        # Format when trips is a dict with $values
        for trip in trips["$values"]:
            if "city" in trip and trip["city"]:
                past_cities.append(trip["city"])
    elif isinstance(trips, list):
        # Format when trips is directly a list
        for trip in trips:
            if "city" in trip and trip["city"]:
                past_cities.append(trip["city"])
    
    # Remove duplicates
    past_cities = list(set(past_cities))
    
    # Prepare user data (handling possible different formats in preferences)
    user_data = {
        'Preferred Places': "",
        'Travel Tags': "",
        'Age': 0,
        'Marital status': "",
        'Children': "",
        'Gender': ""
    }
    
    # Try to extract from preferences, handling different possible formats
    if preferences:
        if isinstance(preferences, dict):
            user_data.update({
                'Preferred Places': preferences.get("preferredPlaces", ""),
                'Travel Tags': preferences.get("travelTags", ""),
                'Age': preferences.get("age", 0),
                'Marital status': preferences.get("maritalStatus", ""),
                'Children': preferences.get("children", ""),
                'Gender': preferences.get("gender", "")
            })
    
    # Add past cities to preferred places
    if past_cities:
        if user_data['Preferred Places']:
            user_data['Preferred Places'] += ", " + ", ".join(past_cities)
        else:
            user_data['Preferred Places'] = ", ".join(past_cities)
    
    return user_data

# Health check endpoint
@app.get(
    "/health", 
    response_model=HealthResponse, 
    tags=["System"],
    summary="Check API health",
    description="Returns the status of the API and whether the recommendation model is trained"
)
async def health_check():
    last_update = "Never"
    if hasattr(recommender, 'recommender') and recommender.recommender:
        last_update = recommender.recommender.get_last_global_update() or "Never"
    
    return {
        "status": "ok",
        "model_trained": recommender.is_trained,
        "last_update": last_update
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
    top_n: int = Query(5, description="Number of recommendations to return", ge=1, le=20)
):
    try:
        logger.info(f"Processing recommendation request for user: {user_id}")
        
        # Get user preferences and trips
        preferences = await get_user_preferences(user_id)
        trips = await get_user_trips(user_id)
        
        # Extract past cities from trips (if any)
        past_cities = []
        
        if isinstance(trips, list):
            past_cities = [trip.get("city", "") for trip in trips if trip.get("city")]
        elif "$values" in trips and trips["$values"]:
            past_cities = [trip.get("city", "") for trip in trips["$values"] if trip.get("city")]
            
        logger.info(f"User has {len(past_cities)} past trip cities: {past_cities}")
        
        # Prepare user data for the recommender
        user_data = prepare_user_data(preferences, trips)
        logger.info(f"Prepared user data for recommendation: {user_data}")
        
        # Get recommendations (exclude already visited places)
        recommendations = recommender.recommend(user_data, top_n=top_n, additional_exclusions=past_cities)
        
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

# New endpoint for direct recommendations without a user account
@app.post(
    "/api/recommendations", 
    response_model=RecommendationResponse,
    responses={
        404: {"model": ErrorResponse, "description": "No recommendations generated"},
        500: {"model": ErrorResponse, "description": "Server error"}
    },
    tags=["Recommendations"],
    summary="Get recommendations based on provided preferences",
    description="Generates travel recommendations based on provided preferences without requiring a user account"
)
async def get_direct_recommendations(
    request: RecommendationRequest,
    top_n: int = Query(5, description="Number of recommendations to return", ge=1, le=20)
):
    try:
        logger.info("Processing direct recommendation request")
        
        # Prepare user data
        user_data = {
            'Preferred Places': request.preferred_places or "",
            'Travel Tags': request.travel_tags or "",
            'Age': request.age or 0,
            'Marital status': request.marital_status or "",
            'Children': request.children or "",
            'Gender': request.gender or ""
        }
        
        logger.info(f"Direct recommendation request data: {user_data}")
        
        # Get recommendations
        recommendations = recommender.recommend(user_data, top_n=top_n)
        
        if not recommendations:
            logger.warning("No direct recommendations generated")
            raise HTTPException(
                status_code=404,
                detail="Could not generate recommendations. Please ensure the model is trained."
            )
        
        # Return recommendations
        response = {
            "success": True,
            "user_id": "direct_request",
            "recommended_places": recommendations['recommendations'],
            "similar_users": recommendations['similar_users'],
            "similarity_scores": recommendations['similarity_scores']
        }
        
        logger.info(f"Generated direct recommendations: {recommendations['recommendations']}")
        return response
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"Error generating direct recommendations: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error generating direct recommendations: {str(e)}"
        )

# New endpoint to update user after trip completion
@app.post(
    "/api/update-user/{user_id}",
    response_model=UserUpdateResponse,
    tags=["Model Management"],
    summary="Update user data after completing a trip",
    description="Updates a user in the recommendation model with new trip data"
)
async def update_user(user_id: str, request: UserUpdateRequest):
    try:
        logger.info(f"Updating user {user_id} with new trip data")
        
        if "city" in request.trip_data and request.trip_data["city"]:
            place_name = request.trip_data["city"]
            
            # Update the user in the recommendation model
            if recommender.add_new_user_trip(user_id=user_id, place_name=place_name):
                current_time = datetime.utcnow().isoformat()
                
                return {
                    "success": True,
                    "message": f"User {user_id} updated with new trip to {place_name}",
                    "updated_at": current_time
                }
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to update user {user_id}"
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="Trip data must include a city name"
            )
    
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"Error updating user: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error updating user: {str(e)}"
        )

# Batch update endpoint
@app.post(
    "/api/batch-update",
    response_model=BatchUpdateResponse,
    tags=["Model Management"],
    summary="Update multiple users at once",
    description="Batch updates multiple users in the recommendation model"
)
async def batch_update_users(request: BatchUpdateRequest):
    try:
        logger.info(f"Batch update request received for {len(request.user_list)} users")
        
        # Process each user in the list
        updated_users = []
        for user_id in request.user_list:
            try:
                # Get user data
                preferences = await get_user_preferences(user_id)
                trips = await get_user_trips(user_id)
                
                # Prepare user data
                user_data = prepare_user_data(preferences, trips)
                
                # Add to update list
                updated_users.append({
                    "user_id": user_id,
                    "user_data": user_data
                })
            except Exception as user_e:
                logger.error(f"Error processing user {user_id}: {str(user_e)}")
        
        # Perform batch update
        if updated_users:
            updated_count = recommender.batch_update_users(updated_users)
            
            return {
                "success": True,
                "updated_count": updated_count,
                "message": f"Successfully updated {updated_count} users"
            }
        else:
            return {
                "success": False,
                "updated_count": 0,
                "message": "No valid users to update"
            }
    
    except Exception as e:
        logger.error(f"Error in batch update: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error in batch update: {str(e)}"
        )

# Model metadata endpoint
@app.get(
    "/api/model-metadata",
    response_model=ModelMetadataResponse,
    tags=["Model Management"],
    summary="Get model metadata",
    description="Returns information about the model update status"
)
async def get_model_metadata():
    try:
        last_update = recommender.recommender.get_last_global_update() or datetime.utcnow().isoformat()
        
        # Get count of users
        user_count = len(recommender.recommender.train_df) if recommender.is_trained else 0
        
        # Get count of users updated in last 24 hours
        yesterday = (datetime.utcnow() - timedelta(hours=24)).isoformat()
        updated_users = recommender.get_recent_active_users(since=yesterday)
        
        # Get total trips processed
        all_trips = await get_all_trips()
        
        return {
            "last_global_update": last_update,
            "user_count": user_count,
            "updated_users_24h": len(updated_users),
            "total_trips_processed": len(all_trips)
        }
    
    except Exception as e:
        logger.error(f"Error getting model metadata: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error getting model metadata: {str(e)}"
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

