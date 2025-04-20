import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Parallel, delayed
import time
import tracemalloc
import pandas as pd

class RecommendationEngine:
    """
    Handles similarity calculations and recommendation generation.
    """
    def __init__(self, batch_size=100, n_jobs=-1):
        """Initialize recommendation engine with batch processing parameters"""
        self.weighted_matrix = None
        self.train_df = None
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.similarity_matrix = None
    
    def fit(self, train_df, weighted_matrix):
        """Fit recommendation engine with training data"""
        self.train_df = train_df
        self.weighted_matrix = weighted_matrix
        return self
    
    def compute_similarity_matrix(self, use_batches=True):
        """
        Compute similarity matrix with batch processing and parallel execution
        """
        print("Computing similarity matrix...")
        start_time = time.time()
        tracemalloc.start()
        
        n_users = self.weighted_matrix.shape[0]
        
        if use_batches:
            # Initialize empty similarity matrix
            self.similarity_matrix = sp.lil_matrix((n_users, n_users))
            
            # Process in batches
            batch_indices = [(i, min(i + self.batch_size, n_users)) 
                            for i in range(0, n_users, self.batch_size)]
            
            for start_idx, end_idx in batch_indices:
                print(f"Processing batch {start_idx} to {end_idx}")
                batch_matrix = self.weighted_matrix[start_idx:end_idx]
                
                # Compute similarities for this batch in parallel
                batch_sim = Parallel(n_jobs=self.n_jobs)(
                    delayed(self._compute_row_similarities)(
                        i, batch_matrix[i-start_idx], self.weighted_matrix
                    )
                    for i in range(start_idx, end_idx)
                )
                
                # Update similarity matrix
                for i, similarities in enumerate(batch_sim):
                    row_idx = start_idx + i
                    # Store only significant similarities (> 0.1) to save memory
                    for j, sim in enumerate(similarities):
                        if sim > 0.1:  # threshold for storing
                            self.similarity_matrix[row_idx, j] = sim
            
            # Convert to CSR for efficient row slicing
            self.similarity_matrix = self.similarity_matrix.tocsr()
        else:
            # For small datasets, compute the full matrix at once
            self.similarity_matrix = cosine_similarity(self.weighted_matrix)
        
        # Memory profiling
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end_time = time.time()
        
        print(f"Similarity computation complete in {end_time - start_time:.2f} seconds")
        print(f"Current memory usage: {current / 10**6:.2f} MB")
        print(f"Peak memory usage: {peak / 10**6:.2f} MB")
        
        return self.similarity_matrix
    
    def _compute_row_similarities(self, row_idx, row_vector, matrix):
        """Compute similarities for a single row"""
        return cosine_similarity(row_vector, matrix)[0]
    
    def recommend_places(self, user_vector=None, user_idx=None, top_n=5, exclude_visited=True):
        """
        Recommend places based on user vector or user index
        
        Parameters:
        - user_vector: Vectorized features of the user (for new users)
        - user_idx: Index of the user in training data (for existing users)
        - top_n: Number of recommendations to return
        - exclude_visited: Whether to exclude places user has already visited
        
        Returns:
        - recommended_places: List of recommended place names
        - similar_users: Indices of most similar users
        - similarity_scores: Similarity scores for those users
        """
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()
        
        # Get similarity scores
        if user_idx is not None:
            # For existing users, get from similarity matrix
            sim_scores = self.similarity_matrix[user_idx].toarray().flatten()
            visited_places = set(self.train_df.iloc[user_idx]['Preferred Places'].split(', '))
        else:
            # For new users, compute similarities
            sim_scores = cosine_similarity(user_vector, self.weighted_matrix)[0]
            visited_places = set()  # Will be filled later
        
        # Find most similar users
        if exclude_visited and user_idx is not None:
            # Exclude self from recommendations
            sim_scores[user_idx] = 0
            
        most_similar_users_indices = sim_scores.argsort()[-top_n*2:][::-1]  # Get more for filtering
        
        # For new users with a string of preferences
        if user_vector is not None and hasattr(user_vector, 'split'):
            # Extract visited places from the preference string
            try:
                visited_places = set(user_vector.split(', ')[:1])
            except (AttributeError, TypeError) as e:
                print(f"Warning: Could not extract visited places from vector: {e}")
                visited_places = set()
        
        # Get place recommendations
        place_counts = {}
        for sim_user_idx in most_similar_users_indices:
            if sim_user_idx < len(self.train_df):
                places = self.train_df.iloc[sim_user_idx]['Preferred Places'].split(', ')
                for place in places:
                    if place and (not exclude_visited or place not in visited_places):
                        place_counts[place] = place_counts.get(place, 0) + sim_scores[sim_user_idx]
        
        # Sort by weighted count (frequency × similarity)
        sorted_places = sorted(place_counts.items(), key=lambda x: x[1], reverse=True)
        recommended_places = [place for place, _ in sorted_places[:top_n]]
        
        return recommended_places, most_similar_users_indices[:top_n], sim_scores[most_similar_users_indices[:top_n]]
    
    
    def update_with_new_user(self, vectorizer, user_data, recompute_similarity=False):
        """
        Update the model with a new user without full retraining
        """
        # Process user data to get combined features
        combined_features = (
            user_data.get('Preferred Places', '') + " " + 
            user_data.get('Travel Tags', '') + " " +
            str(user_data.get('Age', '')) + " " + 
            user_data.get('Marital status', '') + " " +
            user_data.get('Children', '') + " " + 
            user_data.get('Gender', '')
        )
        
        # Transform the new user features
        new_user_vector = vectorizer.transform_new_user(combined_features)
        
        # Extend training data and weighted matrix
        # Use pd.concat instead of append which is deprecated
        self.train_df = pd.concat([self.train_df, pd.DataFrame([user_data])], ignore_index=True)
        self.weighted_matrix = sp.vstack([self.weighted_matrix, new_user_vector])
        
        # Recompute similarity matrix if requested
        if recompute_similarity:
            self.compute_similarity_matrix()
        else:
            # Just compute similarities for the new user
            if self.similarity_matrix is not None:
                # Calculate similarities between new user and all users (including the new one)
                new_similarities = cosine_similarity(new_user_vector, self.weighted_matrix)[0]
                n = self.similarity_matrix.shape[0]
                
                # Create new expanded similarity matrix with correct dimensions
                expanded_matrix = sp.lil_matrix((n+1, n+1))
                
                # Copy existing similarities
                expanded_matrix[:n, :n] = self.similarity_matrix
                
                # Add new user similarities (exclude last element which is similarity to self)
                expanded_matrix[n, :n] = sp.csr_matrix(new_similarities[:n])
                expanded_matrix[:n, n] = sp.csr_matrix(new_similarities[:n]).T
                
                # Self-similarity is 1.0
                expanded_matrix[n, n] = 1.0
                
                # Convert back to CSR format for efficiency
                self.similarity_matrix = expanded_matrix.tocsr()
        
        return len(self.train_df) - 1  # Return new user index
    
    def save(self, path="models"):
        """Save model state"""
        import joblib
        import os
        
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.train_df, f"{path}/train_df.pkl")
        sp.save_npz(f"{path}/weighted_matrix.npz", self.weighted_matrix)
        if self.similarity_matrix is not None:
            sp.save_npz(f"{path}/similarity_matrix.npz", self.similarity_matrix)
        print(f"Recommendation engine saved to {path}")
    
    def load(self, path="models"):
        """Load model state"""
        import joblib
        
        self.train_df = joblib.load(f"{path}/train_df.pkl")
        self.weighted_matrix = sp.load_npz(f"{path}/weighted_matrix.npz")
        try:
            self.similarity_matrix = sp.load_npz(f"{path}/similarity_matrix.npz")
        except:
            self.similarity_matrix = None
        print(f"Recommendation engine loaded from {path}")