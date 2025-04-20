import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split

class DataProcessor:
    """
    Handles data loading, preprocessing, and feature engineering for the recommendation system.
    """
    def __init__(self, category_columns=None):
        """Initialize the data processor with category columns"""
        self.category_columns = category_columns or [
            "Historical Sites", "Beaches", "Adventure", "Nile Cruises",
            "Religious Tourism", "Desert Exploration", "Relaxation"
        ]
    
    def load_data(self, users_path, places_path):
        """Load user and place datasets"""
        try:
            self.users_df = pd.read_excel(users_path)
            self.places_df = pd.read_excel(places_path)
            print(f"Data loaded: {len(self.users_df)} users, {len(self.places_df)} places")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def preprocess_places(self):
        """Process places data and create category information"""
        # Ensure category columns are numeric
        for col in self.category_columns:
            if col in self.places_df.columns:
                self.places_df[col] = self.places_df[col].apply(
                    pd.to_numeric, errors='coerce').fillna(0).astype(int)
            else:
                self.places_df[col] = 0
        
        # Create combined_info column for each place
        self.places_df['combined_info'] = self.places_df[self.category_columns].apply(
            lambda row: ' '.join(row.index[row == 1].tolist()), axis=1)
        
        return self.places_df
    
    def preprocess_users(self):
        """Process user data and merge with place categories"""
        # Ensure users_df has all required columns
        for col in ['User ID', 'Age', 'Gender', 'Marital status', 'Children', 'Travel Tags', 'Preferred Places']:
            if col not in self.users_df.columns:
                self.users_df[col] = ''
        
        # Make sure both DataFrames have the needed columns before merging
        required_columns = ['Place name', 'combined_info'] + self.category_columns
        for col in required_columns:
            if col not in self.places_df.columns:
                if col == 'Place name':
                    raise ValueError("Places data must have 'Place name' column")
                elif col == 'combined_info':
                    self.places_df['combined_info'] = ''
                else:
                    self.places_df[col] = 0
        
        # Reset the users DataFrame with the necessary columns for category information
        temp_df = pd.DataFrame()
        
        # Process each user individually to handle multiple preferred places
        user_rows = []
        
        for _, user in self.users_df.iterrows():
            user_id = user['User ID']
            age = user['Age']
            gender = user['Gender']
            marital_status = user['Marital status']
            children = user['Children']
            travel_tags = user['Travel Tags']
            
            # Handle multiple preferred places
            preferred_places = str(user['Preferred Places']).split(', ')
            
            # For each place, create a row
            for place in preferred_places:
                place = place.strip()
                if not place:  # Skip empty places
                    continue
                
                place_info = self.places_df[self.places_df['Place name'] == place]
                
                if len(place_info) == 0:
                    # Place not found, create a row without category info
                    row_data = {
                        'User ID': user_id,
                        'Age': age,
                        'Gender': gender,
                        'Marital status': marital_status,
                        'Children': children,
                        'Travel Tags': travel_tags,
                        'Preferred Places': place,
                        'combined_info': ''
                    }
                    # Add empty category columns
                    for cat in self.category_columns:
                        row_data[cat] = 0
                    
                else:
                    # Place found, create row with category info
                    place_info = place_info.iloc[0]
                    row_data = {
                        'User ID': user_id,
                        'Age': age,
                        'Gender': gender,
                        'Marital status': marital_status,
                        'Children': children,
                        'Travel Tags': travel_tags,
                        'Preferred Places': place,
                        'combined_info': place_info['combined_info']
                    }
                    # Add category columns
                    for cat in self.category_columns:
                        row_data[cat] = place_info[cat]
                
                user_rows.append(row_data)
        
        # Create new dataframe with all processed rows
        temp_df = pd.DataFrame(user_rows)
        
        # If no rows were created (e.g., all users have empty preferred places),
        # create an empty dataframe with the correct columns
        if len(temp_df) == 0:
            columns = ['User ID', 'Age', 'Gender', 'Marital status', 'Children', 
                      'Travel Tags', 'Preferred Places', 'combined_info'] + self.category_columns
            temp_df = pd.DataFrame(columns=columns)
        
        # Aggregate user preferences
        self.users_df = self._aggregate_user_preferences(temp_df)
        
        return self.users_df
    
    def _aggregate_user_preferences(self, df=None):
        """Aggregate preferences for users with multiple places"""
        if df is None:
            df = self.users_df
        
        # Check if required columns exist
        for col in self.category_columns + ['combined_info']:
            if col not in df.columns:
                df[col] = 0 if col != 'combined_info' else ''
        
        # Group by user attributes and aggregate
        aggregated = df.groupby(['User ID', 'Age', 'Gender', 'Marital status', 'Children', 'Travel Tags']) \
            .agg({
                'Preferred Places': lambda x: ', '.join(filter(None, x)),
                'combined_info': lambda x: ' '.join(filter(None, x)),
                **{col: 'max' for col in self.category_columns}
            }).reset_index()
        
        return aggregated
    
    def create_feature_text(self):
        """Create combined features text for TF-IDF processing"""
        self.users_df['combined_features'] = (
            self.users_df['Preferred Places'].astype(str) + " " + 
            self.users_df['Travel Tags'].astype(str) + " " +
            self.users_df['Age'].astype(str) + " " + 
            self.users_df['Marital status'].astype(str) + " " +
            self.users_df['Children'].astype(str) + " " + 
            self.users_df['Gender'].astype(str) + " " + 
            self.users_df['combined_info'].astype(str)
        )
        return self.users_df
    
    def split_train_test(self, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        return train_test_split(
            self.users_df, test_size=test_size, random_state=random_state
        )
    
    def add_new_place(self, place_data):
        """Add a new place to the places dataset"""
        if place_data['Place name'] in self.places_df['Place name'].values:
            print(f"Place '{place_data['Place name']}' already exists")
            return False
        
        # Ensure the new place has all required columns
        for col in self.category_columns:
            if col not in place_data:
                place_data[col] = 0
        
        # Add new place
        self.places_df = pd.concat([self.places_df, pd.DataFrame([place_data])], ignore_index=True)
        
        # Generate combined_info for the new place
        idx = self.places_df[self.places_df['Place name'] == place_data['Place name']].index[0]
        
        combined_info = []
        for col in self.category_columns:
            if self.places_df.loc[idx, col] == 1:
                combined_info.append(col)
                
        self.places_df.loc[idx, 'combined_info'] = ' '.join(combined_info)
        
        # Save updated places data
        self.places_df.to_excel("Places.xlsx", index=False)
        print(f"Added new place: {place_data['Place name']}")
        return True
    
    def add_new_user_trip(self, user_id, place_name):
        """Update an existing user with a new trip"""
        if user_id not in self.users_df['User ID'].values:
            print(f"User ID {user_id} not found")
            return False
            
        if place_name not in self.places_df['Place name'].values:
            print(f"Place '{place_name}' not found")
            return False
        
        # Get current preferred places
        user_idx = self.users_df[self.users_df['User ID'] == user_id].index[0]
        current_places = self.users_df.loc[user_idx, 'Preferred Places']
        
        # Check if place is already in preferred places
        if isinstance(current_places, str) and place_name in current_places.split(', '):
            print(f"User already has {place_name} in their preferred places")
            return False
        
        # Update preferred places
        updated_places = f"{current_places}, {place_name}" if current_places else place_name
        
        # Store the user's current information
        user_data = self.users_df.loc[user_idx].copy()
        user_data['Preferred Places'] = updated_places
        
        # Remove the user from the DataFrame
        self.users_df = self.users_df[self.users_df['User ID'] != user_id]
        
        # Create a temporary DataFrame with the updated user
        temp_df = pd.DataFrame([user_data])
        
        # Reload the entire dataset to ensure proper merging with places
        # This is a more reliable approach than trying to patch the existing DataFrame
        self.users_df = pd.concat([self.users_df, temp_df], ignore_index=True)
        
        # Save updated users data
        self.users_df.to_excel("Users.xlsx", index=False)
        print(f"Added trip to {place_name} for User ID {user_id}")
            
        # Now reprocess everything to ensure consistency
        self.preprocess_places()  # Ensure all places are properly processed
        self.preprocess_users()   # This will now correctly incorporate the new trip
        self.create_feature_text()
        
        return True