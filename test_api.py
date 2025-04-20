import requests
import json
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:5050"
USER_ID = "e14dcd98-72cc-4322-9e3e-a91793e0f221"  # The user ID to test
TOP_N = 5  # Number of recommendations to request

def test_recommendations():
    """Test the recommendation API endpoint for a specific user"""
    print(f"=== Testing Recommendation API for User ID: {USER_ID} ===")
    print(f"Current Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"API Base URL: {API_BASE_URL}")
    
    # First check the health endpoint
    try:
        health_response = requests.get(f"{API_BASE_URL}/health")
        health_data = health_response.json()
        print("\n== API Health Check ==")
        print(f"Status: {health_response.status_code}")
        print(f"API status: {health_data.get('status')}")
        print(f"Model trained: {health_data.get('model_trained')}")
        
        if health_data.get('model_trained') is not True:
            print("WARNING: Model is not trained. Recommendations may not work correctly.")
    except Exception as e:
        print(f"Error checking API health: {str(e)}")
        return
    
    # Get recommendations
    try:
        print(f"\n== Getting Recommendations for User: {USER_ID} ==")
        start_time = datetime.now()
        
        endpoint = f"{API_BASE_URL}/api/recommendations/{USER_ID}?top_n={TOP_N}"
        print(f"Request URL: {endpoint}")
        
        response = requests.get(endpoint)
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
        print(f"Response Status: {response.status_code}")
        print(f"Response Time: {elapsed_time:.3f} seconds")
        
        # Process results
        if response.status_code == 200:
            data = response.json()
            print("\n== Recommendation Results ==")
            print(f"Success: {data.get('success')}")
            print(f"User ID: {data.get('user_id')}")
            
            # Print recommended places
            recommended_places = data.get('recommended_places', [])
            print(f"\nTop {len(recommended_places)} Recommended Places:")
            for i, place in enumerate(recommended_places, 1):
                print(f"{i}. {place}")
            
            # Print similar users (just the count)
            similar_users = data.get('similar_users', [])
            print(f"\nSimilar Users: {len(similar_users)} found")
            
            # Print full response for debugging
            print("\n== Full JSON Response ==")
            print(json.dumps(data, indent=2))
            
            return data
        else:
            print("Error getting recommendations:")
            print(json.dumps(response.json(), indent=2))
            return None
            
    except Exception as e:
        print(f"Error during API test: {str(e)}")
        return None

if __name__ == "__main__":
    test_recommendations()