import os
import pandas as pd
import numpy as np
import tracemalloc
import time
from datetime import datetime, timedelta
import json
from data_processor import DataProcessor
from vectorizer import FeatureVectorizer
from recommender import RecommendationEngine
from evaluator import ModelEvaluator
from memory_profiler import MemoryProfiler

class TravelRecommender:
    """
    Main class for travel recommendation system.
    
    Integrates data processing, feature vectorization, 
    recommendation engine, and model evaluation components.
    """
    def __init__(self, data_dir="data", model_dir="models"):
        """Initialize travel recommender with directories for data and models"""
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.data_processor = DataProcessor(data_dir)
        self.vectorizer = FeatureVectorizer()
        self.recommender = RecommendationEngine()
        self.evaluator = ModelEvaluator()
        self.memory_profiler = MemoryProfiler()
        self.is_trained = False
        
        # Create directories if they don't exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        # Load or create metadata
        self.metadata_path = os.path.join(model_dir, "metadata.json")
        self.load_metadata()
    
    def train(self, users_path=None, places_path=None, retrain=False):
        """
        Train the recommendation model
        
        Parameters:
        - users_path: Path to users data file
        - places_path: Path to places data file
        - retrain: Whether to force retraining
        """
        with self.memory_profiler.profile_block("Training"):
            if users_path is None:
                users_path = os.path.join(self.data_dir, "Users.xlsx")
            if places_path is None:
                places_path = os.path.join(self.data_dir, "Places.xlsx")
            
            # Check if model already exists and we're not forcing retraining
            if os.path.exists(os.path.join(self.model_dir, "train_df.pkl")) and not retrain:
                print("Loading existing model...")
                self.load_model()
                return True
            
            print("Training new model...")
            
            # Load and preprocess data
            self.data_processor.load_data(users_path, places_path)
            self.data_processor.preprocess_places()
            self.data_processor.preprocess_users()
            self.data_processor.create_feature_text()
            
            # Split into train and test sets
            train_df, test_df = self.data_processor.split_train_test()
            
            # Vectorize features
            weighted_matrix_train = self.vectorizer.fit_transform(train_df)
            
            # Train recommender
            self.recommender.fit(train_df, weighted_matrix_train)
            
            # Compute similarity matrix
            self.recommender.compute_similarity_matrix()
            
            # Evaluate model
            self.evaluator.evaluate(self.recommender, self.vectorizer, test_df)
            
            # Initialize user metadata with current time
            self.initialize_user_metadata()
            
            # Save model
            self.save_model()
            
            self.is_trained = True
            return True
    
    def initialize_user_metadata(self):
        """Initialize metadata for all users with current timestamp"""
        if hasattr(self.recommender, 'train_df') and self.recommender.train_df is not None:
            if 'User ID' in self.recommender.train_df.columns:
                now = datetime.utcnow().isoformat()
                for user_id in self.recommender.train_df['User ID']:
                    self.recommender.user_metadata[str(user_id)] = now
                
                # Update global timestamp
                self.recommender.update_global_timestamp()
    
    def recommend(self, user_data, top_n=5, additional_exclusions=None):
        """
        Get recommendations for a user
        
        Parameters:
        - user_data: Either a user ID or a dictionary with user preferences
        - top_n: Number of recommendations to return
        - additional_exclusions: Additional places to exclude from recommendations
        """
        if not self.is_trained:
            print("Model not trained. Please train the model first.")
            return None
        
        # If user_data is an ID, look up the user
        if isinstance(user_data, (int, np.integer)) or (isinstance(user_data, str) and user_data.isdigit()):
            user_id = str(user_data)
            user_idx = None
            
            # Find user index by ID
            if hasattr(self.recommender, 'user_index_mapping'):
                if user_id in self.recommender.user_index_mapping:
                    user_idx = self.recommender.user_index_mapping[user_id]
                elif int(user_id) in self.recommender.user_index_mapping:
                    user_idx = self.recommender.user_index_mapping[int(user_id)]
            
            if user_idx is None:
                for i, idx in enumerate(self.recommender.train_df['User ID']):
                    if str(idx) == user_id:
                        user_idx = i
                        break
            
            if user_idx is None or user_idx >= len(self.recommender.train_df):
                print(f"User ID {user_id} not found.")
                return None
            
            recommendations, similar_users, similarity_scores = self.recommender.recommend_places(
                user_idx=user_idx, 
                user_id=user_id,
                top_n=top_n,
                additional_exclusions=additional_exclusions
            )
        else:
            # For a new user with preferences
            if 'combined_features' not in user_data:
                # Create combined features
                combined_features = (
                    user_data.get('Preferred Places', '') + " " + 
                    user_data.get('Travel Tags', '') + " " +
                    str(user_data.get('Age', '')) + " " + 
                    user_data.get('Marital status', '') + " " +
                    user_data.get('Children', '') + " " + 
                    user_data.get('Gender', '')
                )
            else:
                combined_features = user_data['combined_features']
            
            # Get user vector
            user_vector = self.vectorizer.transform_new_user(combined_features)
            
            # Get recommendations
            recommendations, similar_users, similarity_scores = self.recommender.recommend_places(
                user_vector=user_vector, 
                top_n=top_n,
                additional_exclusions=additional_exclusions
            )
        
        return {
            'recommendations': recommendations,
            'similar_users': similar_users.tolist() if hasattr(similar_users, 'tolist') else similar_users,
            'similarity_scores': similarity_scores.tolist() if hasattr(similarity_scores, 'tolist') else similarity_scores
        }
    
    def add_new_place(self, place_data):
        """Add a new place to the system"""
        with self.memory_profiler.profile_block("Adding New Place"):
            success = self.data_processor.add_new_place(place_data)
            if success and self.is_trained:
                # We don't need to retrain the whole model since places
                # only affect the system when users visit them
                print("Place added successfully. Model will update when users visit this place.")
            return success
    
    def add_new_user_trip(self, user_id=None, user_data=None, place_name=None):
        """
        Update user with a new trip or add a completely new user
        
        Parameters:
        - user_id: ID of existing user to update
        - user_data: Complete data for a new user
        - place_name: Place the user visited
        """
        with self.memory_profiler.profile_block("Adding User Trip"):
            if user_id is not None and place_name is not None:
                # Updating existing user
                success = self.data_processor.add_new_user_trip(user_id, place_name)
                if success and self.is_trained:
                    print("Updating model with new trip data...")
                    
                    # After data_processor has updated the user, find the updated user data
                    user_row = self.data_processor.users_df[self.data_processor.users_df['User ID'] == user_id]
                    if len(user_row) == 0:
                        print(f"Error: User {user_id} not found in updated data")
                        return False
                    
                    # Convert user row to dict
                    user_data_dict = user_row.iloc[0].to_dict()
                    
                    # Update user in the recommendation model
                    self.recommender.update_user(
                        user_id=user_id, 
                        new_user_data=user_data_dict, 
                        vectorizer=self.vectorizer
                    )
                    
                    # Update metadata timestamp for this user
                    self.recommender.user_metadata[str(user_id)] = datetime.utcnow().isoformat()
                    
                    # Save updated model
                    self.save_model()
                    
                return success
            
            elif user_data is not None:
                # Adding new user
                if 'User ID' not in user_data:
                    # Generate a new user ID
                    max_id = self.data_processor.users_df['User ID'].max()
                    user_data['User ID'] = max_id + 1
                
                # Also update the raw_users_df
                if self.data_processor.raw_users_df is None:
                    raw_columns = ['User ID', 'Age', 'Gender', 'Marital status', 'Children', 'Travel Tags', 'Preferred Places']
                    self.data_processor.raw_users_df = self.data_processor.users_df[raw_columns].copy()
                    
                # Add raw user data
                raw_user_data = {col: user_data.get(col, '') for col in 
                            ['User ID', 'Age', 'Gender', 'Marital status', 'Children', 'Travel Tags', 'Preferred Places']}
                self.data_processor.raw_users_df = pd.concat([self.data_processor.raw_users_df, pd.DataFrame([raw_user_data])], ignore_index=True)
                
                # Save raw user data
                self.data_processor.raw_users_df.to_excel(os.path.join(self.data_dir, "Users.xlsx"), index=False)
                
                # Add to the processed users dataframe
                self.data_processor.users_df = pd.concat([self.data_processor.users_df, pd.DataFrame([user_data])], ignore_index=True)
                
                # Process new user
                self.data_processor.preprocess_users()
                self.data_processor.create_feature_text()
                
                if self.is_trained:
                    # Update model with new user
                    new_user_idx = self.recommender.update_with_new_user(self.vectorizer, user_data)
                    
                    # Update metadata timestamp for this user
                    self.recommender.user_metadata[str(user_data['User ID'])] = datetime.utcnow().isoformat()
                    
                    print(f"Added new user with ID {user_data['User ID']}, index {new_user_idx}")
                    
                    # Save model
                    self.save_model()
                    
                return True
            
            else:
                print("Either user_id and place_name, or user_data must be provided.")
                return False
    
    def batch_update_users(self, user_list):
        """
        Update multiple users in batch
        
        Parameters:
        - user_list: List of dicts with user_id and user_data to update
        
        Returns:
        - Number of users successfully updated
        """
        if not self.is_trained:
            print("Model not trained. Please train the model first.")
            return 0
        
        updated_count = 0
        for user_item in user_list:
            user_id = user_item.get('user_id')
            user_data = user_item.get('user_data')
            
            if user_id and user_data:
                # Convert user ID to string for consistency
                user_id = str(user_id)
                
                # Update user in recommender
                if self.recommender.update_user(user_id, user_data, self.vectorizer):
                    updated_count += 1
                    
                    # Update timestamp in metadata
                    self.recommender.user_metadata[user_id] = datetime.utcnow().isoformat()
        
        # Save model if any users were updated
        if updated_count > 0:
            self.recommender.update_global_timestamp()
            self.save_model()
        
        return updated_count

    def save_model(self):
        """Save model artifacts"""
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Save processed data
        self.data_processor.save_processed_data()
        
        # Save vectorizer
        self.vectorizer.save(self.model_dir)
        
        # Save recommender (includes metadata)
        self.recommender.save(self.model_dir)
        
        print(f"Model saved to {self.model_dir}")
        
    def load_model(self):
        """Load model artifacts"""
        try:
            # Load vectorizer
            self.vectorizer.load(self.model_dir)
            
            # Load recommender (includes metadata)
            self.recommender.load(self.model_dir)
            
            # Load processed data
            self.data_processor.users_df = pd.read_excel(os.path.join(self.data_dir, "Users_processed.xlsx"))
            self.data_processor.places_df = pd.read_excel(os.path.join(self.data_dir, "Places_processed.xlsx"))
            
            self.is_trained = True
            print("Model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def load_metadata(self):
        """Load metadata from file or create new if not exists"""
        try:
            with open(self.metadata_path, "r") as f:
                self.metadata = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Create default metadata
            self.metadata = {
                "last_global_update": datetime.utcnow().isoformat(),
                "users": {},
                "model_version": "1.0",
                "total_updates": 0
            }
            # Save default metadata
            self.save_metadata()
    
    def save_metadata(self):
        """Save metadata to file"""
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
    
    def get_recent_active_users(self, since=None):
        """
        Get users active since a specific time
        
        Parameters:
        - since: ISO format datetime string or datetime object
        
        Returns:
        - List of user IDs
        """
        if since is None:
            # Default to 24 hours ago
            since = (datetime.utcnow() - timedelta(hours=24)).isoformat()
        elif isinstance(since, datetime):
            since = since.isoformat()
            
        # Convert since to datetime
        since_dt = datetime.fromisoformat(since)
        
        active_users = []
        for user_id, last_updated in self.recommender.user_metadata.items():
            try:
                user_dt = datetime.fromisoformat(last_updated)
                if user_dt > since_dt:
                    active_users.append(user_id)
            except (ValueError, TypeError):
                # Skip users with invalid timestamps
                continue
        
        return active_users
    
    def get_memory_profile(self):
        """Get memory profiling summary"""
        return self.memory_profiler.get_summary()
    
    def profile_components(self):
        """Profile memory usage of individual components"""
        import sys
        
        print("\nComponent Memory Usage:")
        
        # Recommender
        if hasattr(self.recommender, 'train_df'):
            train_df_size = sys.getsizeof(self.recommender.train_df)
            print(f"Training DataFrame: {train_df_size / 1024 / 1024:.2f} MB")
        
        if hasattr(self.recommender, 'weighted_matrix'):
            weighted_matrix_size = self.recommender.weighted_matrix.data.nbytes
            print(f"Weighted Matrix: {weighted_matrix_size / 1024 / 1024:.2f} MB")
            
        if hasattr(self.recommender, 'similarity_matrix') and self.recommender.similarity_matrix is not None:
            similarity_matrix_size = self.recommender.similarity_matrix.data.nbytes
            print(f"Similarity Matrix: {similarity_matrix_size / 1024 / 1024:.2f} MB")
        
        # Vectorizer
        if hasattr(self.vectorizer, 'tf') and self.vectorizer.tf is not None:
            vectorizer_size = sys.getsizeof(self.vectorizer.tf)
            print(f"TF-IDF Vectorizer: {vectorizer_size / 1024 / 1024:.2f} MB")