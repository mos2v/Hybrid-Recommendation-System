import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureVectorizer:
    """
    Handles TF-IDF vectorization and feature weighting for recommendation system.
    """
    def __init__(self):
        """Initialize the feature vectorizer"""
        self.tf = None
        self.weight_vector = None
        self.feature_names = None
    
    def fit_transform(self, train_df, feature_weights=None):
        """Fit TF-IDF vectorizer and transform training data with weighting"""
        # Default feature weights if not provided
        if feature_weights is None:
            self.feature_weights = {
                'Preferred Places': 6,
                'Travel Tags': 4,
                'Age': 1,
                'Marital status': 1,
                'Children': 1,
                'Gender': 1,
                'combined_info': 5
            }
        else:
            self.feature_weights = feature_weights
        
        # Create and fit TF-IDF vectorizer
        self.tf = TfidfVectorizer(stop_words='english', use_idf=False)
        tf_matrix = self.tf.fit_transform(train_df['combined_features'])
        
        # Get feature names and create weight vector
        self.feature_names = self.tf.get_feature_names_out()
        self.weight_vector = self._create_weight_vector(train_df)
        
        # Apply weights while keeping the matrix sparse
        weighted_matrix = self._apply_weights_sparse(tf_matrix)
        
        return weighted_matrix
    
    def transform(self, test_df):
        """Transform test data using fitted vectorizer"""
        if self.tf is None:
            raise ValueError("Vectorizer not fitted. Call fit_transform first.")
        
        tf_matrix = self.tf.transform(test_df['combined_features'])
        weighted_matrix = self._apply_weights_sparse(tf_matrix)
        
        return weighted_matrix
    
    def _create_weight_vector(self, train_df):
        """Create weight vector based on TF-IDF features"""
        weight_vector = np.ones(len(self.feature_names))
        
        # Helper function to check if a feature belongs to a specific column
        def feature_in_column(feature, column_values, as_str=True):
            """
            Check if a feature appears in any of the values in column_values
            
            Parameters:
            - feature: Feature name as string
            - column_values: Series of column values to check for feature
            - as_str: Whether to convert column values to strings (default True)
            
            Returns:
            - Boolean indicating if feature appears in any column value
            """
            # Handle NaN values and convert to string if needed
            column_values = column_values.fillna('')
            if as_str:
                column_values = column_values.astype(str)
            
            # Extract unique words from column values
            words = set()
            for value in column_values:
                if isinstance(value, str) and value.strip():
                    words.update(word.lower() for word in value.split())
            
            # Check if feature is in any of the words
            return any(feature == word for word in words)
        
        # Assign weights based on feature origins
        for i, feature in enumerate(self.feature_names):
            if feature_in_column(feature, train_df['Preferred Places']):
                weight_vector[i] = self.feature_weights['Preferred Places']
            elif feature_in_column(feature, train_df['Travel Tags']):
                weight_vector[i] = self.feature_weights['Travel Tags']
            elif feature_in_column(feature, train_df['Age'], as_str=True):
                weight_vector[i] = self.feature_weights['Age']
            elif feature_in_column(feature, train_df['Marital status']):
                weight_vector[i] = self.feature_weights['Marital status']
            elif feature_in_column(feature, train_df['Children']):
                weight_vector[i] = self.feature_weights['Children']
            elif feature_in_column(feature, train_df['Gender']):
                weight_vector[i] = self.feature_weights['Gender']
            elif feature_in_column(feature, train_df['combined_info']):
                weight_vector[i] = self.feature_weights['combined_info']
                
        return weight_vector
    
    def _apply_weights_sparse(self, tf_matrix):
        """Apply weights to TF-IDF matrix while keeping it sparse"""
        # Create diagonal weight matrix
        weight_diag = sp.diags(self.weight_vector)
        
        # Multiply sparse matrices
        weighted_matrix = tf_matrix.dot(weight_diag)
        
        return weighted_matrix
    
    def transform_new_user(self, user_features):
        """Transform a single new user's features"""
        if self.tf is None:
            raise ValueError("Vectorizer not fitted. Call fit_transform first.")
        
        # Transform user features
        user_tf = self.tf.transform([user_features])
        
        # Apply weights
        weighted_user_tf = self._apply_weights_sparse(user_tf)
        
        return weighted_user_tf
    
    def save(self, path="models"):
        """Save vectorizer state"""
        import joblib
        import os
        
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.tf, f"{path}/tfidf_vectorizer.pkl")
        np.save(f"{path}/weight_vector.npy", self.weight_vector)
        print(f"Vectorizer saved to {path}")
    
    def load(self, path="models"):
        """Load vectorizer state"""
        import joblib
        
        self.tf = joblib.load(f"{path}/tfidf_vectorizer.pkl")
        self.weight_vector = np.load(f"{path}/weight_vector.npy")
        self.feature_names = self.tf.get_feature_names_out()
        print(f"Vectorizer loaded from {path}")