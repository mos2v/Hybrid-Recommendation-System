import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

class ModelEvaluator:
    """
    Evaluates recommendation system performance with various metrics.
    """
    def __init__(self):
        """Initialize the model evaluator"""
        pass
    
    def evaluate(self, recommender, vectorizer, test_df, top_n=5):
        """Evaluate model with precision, recall, and F1 score"""
        print(f"Evaluating model on {len(test_df)} test users...")
        
        y_true = []
        y_pred = []
        
        for _, row in test_df.iterrows():
            # Get user's actual preferred places
            actual_places = set(row['Preferred Places'].split(", "))
            
            # Get recommendations for this user
            user_vector = vectorizer.transform_new_user(row['combined_features'])
            recommended_places, _, _ = recommender.recommend_places(
                user_vector=user_vector, top_n=top_n
            )
            
            # Calculate true positives and false positives
            y_true.extend([1 if place in actual_places else 0 for place in recommended_places])
            y_pred.extend([1] * len(recommended_places))
        
        # Calculate metrics
        precision = precision_score(y_true, y_pred) if y_pred else 0
        recall = recall_score(y_true, y_pred) if y_pred else 0
        f1 = f1_score(y_true, y_pred) if y_pred else 0
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        
        return metrics
    
    def evaluate_user(self, recommender, vectorizer, user_data, top_n=5):
        """Evaluate recommendations for a specific user"""
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
        user_vector = vectorizer.transform_new_user(combined_features)
        
        # Get recommendations
        recommended_places, similar_users, similarity_scores = recommender.recommend_places(
            user_vector=user_vector, top_n=top_n
        )
        
        return {
            'recommended_places': recommended_places,
            'similar_users': similar_users,
            'similarity_scores': similarity_scores
        }