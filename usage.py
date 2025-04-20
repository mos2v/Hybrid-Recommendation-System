from travel_recommender import TravelRecommender
import os
import tracemalloc
import time

def main():
    # Initialize recommender
    recommender = TravelRecommender()
    
    # Train model
    recommender.train()
    
    # Get recommendations for an existing user
    user_id = 5  # Example user ID
    print(f"\nRecommendations for existing User ID {user_id}:")
    results = recommender.recommend(user_id, top_n=5)
    print(f"Recommended places: {results['recommendations']}")
    print(f"Most similar users: {results['similar_users']}")
    print(f"Similarity scores: {results['similarity_scores']}")
    
    # Get recommendations for a new user
    new_user = {
        'Age': 28,
        'Gender': 'Female',
        'Marital status': 'Single',
        'Children': 'No',
        'Travel Tags': 'Adventure, Relaxation',
        'Preferred Places': 'Cairo'
    }
    print(f"\nRecommendations for new user:")
    results = recommender.recommend(new_user, top_n=5)
    print(f"Recommended places: {results['recommendations']}")
    
    # Add a new place
    new_place = {
        'Place name': 'Mountain Resort',
        'Historical Sites': 0,
        'Beaches': 0,
        'Adventure': 1,
        'Nile Cruises': 0,
        'Religious Tourism': 0,
        'Desert Exploration': 0,
        'Relaxation': 1
    }
    recommender.add_new_place(new_place)
    
    # Add a new trip for an existing user
    recommender.add_new_user_trip(user_id=user_id, place_name='Mountain Resort')
    
    # See how recommendations changed
    print(f"\nRecommendations for User ID {user_id} after new trip:")
    results = recommender.recommend(user_id, top_n=5)
    print(f"Recommended places: {results['recommendations']}")
    
    # Add a completely new user
    new_user_complete = {
            'Age': 35,
            'Gender': 'Male',
            'Marital status': 'Married',
            'Children': 'Yes',
            'Travel Tags': 'Cultural, Historical Sites',
            'Preferred Places': 'Luxor',
        }
    recommender.add_new_user_trip(user_data=new_user_complete)
    
    # Check memory profile
    print("\nMemory Profile:")
    print(recommender.get_memory_profile())
    
    # Profile specific components
    print("\nComponent Memory Usage:")
    recommender.profile_components()

if __name__ == "__main__":
    # Track overall memory usage
    tracemalloc.start()
    start_time = time.time()
    
    main()
    
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"\nOverall Performance:")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    print(f"Current memory usage: {current / 10**6:.2f} MB")
    print(f"Peak memory usage: {peak / 10**6:.2f} MB")