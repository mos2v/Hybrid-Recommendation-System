import pandas as pd
import numpy as np
import joblib
import os
import tracemalloc
import time

class TravelRecommender:
    """
    Main class for the Travel Recommendation System.
    Integrates all components and provides a high-level interface.
    """
    def __init__(self, data_dir="data", model_dir="models"):
        """Initialize the travel recommendation system"""
        from data_processor import DataProcessor
        from vectorizer import FeatureVectorizer
        from recommender import RecommendationEngine
        from evaluator import ModelEvaluator
        from memory_profiler import MemoryProfiler
        
        self.data_dir = data_dir
        self.model_dir = model_dir
        
        # Create directories if they don't exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize components
        self.data_processor = DataProcessor()
        self.vectorizer = FeatureVectorizer()
        self.recommender = RecommendationEngine(batch_size=50)
        self.evaluator = ModelEvaluator()
        self.memory_profiler = MemoryProfiler()
        
        self.is_trained = False
        
    def train(self, users_path=None, places_path=None, retrain=False):
        """Train the recommendation system"""
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
            
            # Save model
            self.save_model()
            
            self.is_trained = True
            return True
    
    def recommend(self, user_data, top_n=5):
        """
        Get recommendations for a user
        
        Parameters:
        - user_data: Either a user ID or a dictionary with user preferences
        - top_n: Number of recommendations to return
        """
        if not self.is_trained:
            print("Model not trained. Please train the model first.")
            return None
        
        # If user_data is an ID, look up the user
        if isinstance(user_data, (int, np.integer)):
            user_idx = user_data
            if user_idx >= len(self.recommender.train_df):
                print(f"User ID {user_idx} not found.")
                return None
            
            recommendations, similar_users, similarity_scores = self.recommender.recommend_places(
                user_idx=user_idx, top_n=top_n
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
                user_vector=user_vector, top_n=top_n
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
                    
                    # Get the user's index in the training data
                    user_indices = self.recommender.train_df[self.recommender.train_df['User ID'] == user_id].index
                    if len(user_indices) == 0:
                        print(f"Error: User {user_id} not found in training data")
                        return False
                        
                    user_idx = user_indices[0]
                    
                    # Create updated feature vector
                    user_vector = self.vectorizer.transform_new_user(user_row['combined_features'].values[0])
                    
                    # Update the matrix at this user's position
                    self.recommender.weighted_matrix[user_idx] = user_vector
                    
                    # Recompute similarities for this user
                    if self.recommender.similarity_matrix is not None:
                        from sklearn.metrics.pairwise import cosine_similarity
                        new_similarities = cosine_similarity(user_vector, self.recommender.weighted_matrix)[0]
                        self.recommender.similarity_matrix[user_idx] = sp.csr_matrix(new_similarities)
                        # Update column (similarities of other users to this user)
                        # We need to convert to lil_matrix for efficient column updates
                        sim_matrix_lil = self.recommender.similarity_matrix.tolil()
                        sim_matrix_lil[:, user_idx] = sp.lil_matrix(new_similarities).T
                        self.recommender.similarity_matrix = sim_matrix_lil.tocsr()
                    
                    # Save updated model
                    self.save_model()
                    
                return success
            
                
            elif user_data is not None:
                # Adding new user
                if 'User ID' not in user_data:
                    # Generate a new user ID
                    max_id = self.data_processor.users_df['User ID'].max()
                    user_data['User ID'] = max_id + 1
                
                # Add to the users dataframe
                self.data_processor.users_df = pd.concat([self.data_processor.users_df, pd.DataFrame([user_data])], ignore_index=True)
                
                # Process new user
                self.data_processor.preprocess_users()
                self.data_processor.create_feature_text()
                
                if self.is_trained:
                    # Update model with new user
                    new_user_idx = self.recommender.update_with_new_user(self.vectorizer, user_data)
                    print(f"Added new user with ID {user_data['User ID']}, index {new_user_idx}")
                    
                    # Save updated users data and model
                    self.data_processor.users_df.to_excel(os.path.join(self.data_dir, "Users.xlsx"), index=False)
                    self.save_model()
                    
                return True
            
            else:
                print("Either user_id and place_name, or user_data must be provided.")
                return False

    def save_model(self):
        """Save model artifacts"""
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Save processed data
        pd.to_excel = self.data_processor.users_df.to_excel(os.path.join(self.data_dir, "Users_processed.xlsx"), index=False)
        pd.to_excel = self.data_processor.places_df.to_excel(os.path.join(self.data_dir, "Places_processed.xlsx"), index=False)
        
        # Save vectorizer
        self.vectorizer.save(self.model_dir)
        
        # Save recommender
        self.recommender.save(self.model_dir)
        
        print(f"Model saved to {self.model_dir}")
    
    def load_model(self):
        """Load model artifacts"""
        try:
            # Load vectorizer
            self.vectorizer.load(self.model_dir)
            
            # Load recommender
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
    
    def get_memory_profile(self):
        """Get memory profiling summary"""
        return self.memory_profiler.get_summary()
    
    def profile_components(self):
        """Profile memory usage of major components"""
        results = []
        
        # Profile vectorizer
        results.append(self.memory_profiler.profile_object(self.vectorizer.tf, "TF-IDF Vectorizer"))
        results.append(self.memory_profiler.profile_object(self.vectorizer.weight_vector, "Weight Vector"))
        
        # Profile recommendation engine
        results.append(self.memory_profiler.profile_object(self.recommender.weighted_matrix, "Weighted Matrix"))
        results.append(self.memory_profiler.profile_object(self.recommender.similarity_matrix, "Similarity Matrix"))
        results.append(self.memory_profiler.profile_object(self.recommender.train_df, "Training DataFrame"))
        
        return results