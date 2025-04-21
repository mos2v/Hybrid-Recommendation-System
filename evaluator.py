class ModelEvaluator:
    """
    Handles evaluation of the recommendation model.
    """
    def __init__(self):
        """Initialize model evaluator"""
        self.metrics = {}
    
    def evaluate(self, recommender, vectorizer, test_df):
        """
        Evaluate model performance
        
        Parameters:
        - recommender: Trained RecommendationEngine instance
        - vectorizer: Fitted FeatureVectorizer instance
        - test_df: Test dataframe for evaluation
        """
        import numpy as np
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        print("Evaluating model...")
        
        y_true = []
        y_pred = []
        
        # For each user in test set
        for _, row in test_df.iterrows():
            # Get actual preferred places
            actual_places = set(row['Preferred Places'].split(', '))
            
            # Get combined features for this user
            combined_features = row['combined_features']
            
            # Get user vector
            user_vector = vectorizer.transform_new_user(combined_features)
            
            # Get recommendations
            recommended_places, _, _ = recommender.recommend_places(
                user_vector=user_vector, 
                top_n=5
            )
            
            # Compare predictions to actual places
            y_true.extend([1 if place in actual_places else 0 for place in recommended_places])
            y_pred.extend([1] * len(recommended_places))
        
        # Calculate metrics
        if len(y_pred) > 0:
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
        else:
            precision = recall = f1 = 0.0
        
        self.metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        print(f"Evaluation metrics: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        return self.metrics