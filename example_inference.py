#!/usr/bin/env python3
"""
Example script showing how to use the news recommendation inference system.
"""

from inference import NewsRecommendationInference
import json

def main():
    # Initialize the inference system
    print("Initializing Co_NAML_LSTUR inference system...")
    inferencer = NewsRecommendationInference(model_name="Co_NAML_LSTUR")
    
    # Load news data and precompute vectors
    print("Loading news data...")
    inferencer.load_news_data("./data/test/news_parsed.tsv")
    
    print("Precomputing news vectors...")
    inferencer.precompute_news_vectors()
    
    # Example 1: Single user recommendation
    print("\n" + "="*50)
    print("Example 1: Single User Recommendation")
    print("="*50)
    
    user_id = 12345
    clicked_news = ["N37378", "N14827", "N50398"]  # User's reading history
    candidate_news = ["N37378", "N14827", "N50398", "N48265", "N42793", "N20404", "N55314"]
    
    print(f"User ID: {user_id}")
    print(f"User's reading history: {clicked_news}")
    print(f"Candidate news: {candidate_news}")
    
    # Get top-5 recommendations
    recommendations = inferencer.recommend_top_k(
        user_id=user_id,
        clicked_news_ids=clicked_news,
        candidate_news_ids=candidate_news,
        k=5
    )
    
    print(f"\nTop 5 recommendations:")
    for i, (news_id, prob) in enumerate(recommendations, 1):
        print(f"{i}. News {news_id}: {prob:.4f}")
    
    # Example 2: Batch prediction for multiple users
    print("\n" + "="*50)
    print("Example 2: Batch Prediction")
    print("="*50)
    
    # Prepare batch data
    user_data_list = [
        {
            'user_id': 12345,
            'clicked_news_ids': ["N37378", "N14827", "N50398"]
        },
        {
            'user_id': 12346,
            'clicked_news_ids': ["N48265", "N42793", "N20404"]
        },
        {
            'user_id': 12347,
            'clicked_news_ids': ["N55314", "N37378", "N14827"]
        }
    ]
    
    candidate_news_batch = ["N37378", "N14827", "N50398", "N48265", "N42793"]
    
    print(f"Processing {len(user_data_list)} users...")
    print(f"Candidate news: {candidate_news_batch}")
    
    # Run batch prediction
    batch_results = inferencer.batch_predict(user_data_list, candidate_news_batch)
    
    # Display results
    for user_id, predictions in batch_results.items():
        print(f"\nUser {user_id} top 3 recommendations:")
        for i, (news_id, prob) in enumerate(predictions[:3], 1):
            print(f"  {i}. News {news_id}: {prob:.4f}")
    
    # Example 3: Get user vector for similarity analysis
    print("\n" + "="*50)
    print("Example 3: User Vector Analysis")
    print("="*50)
    
    user_vector = inferencer.get_user_vector(user_id=12345, clicked_news_ids=clicked_news)
    print(f"User vector shape: {user_vector.shape}")
    print(f"User vector norm: {user_vector.norm():.4f}")
    
    # Compare with another user
    user_vector_2 = inferencer.get_user_vector(user_id=12346, clicked_news_ids=["N48265", "N42793"])
    similarity = torch.cosine_similarity(user_vector.unsqueeze(0), user_vector_2.unsqueeze(0))
    print(f"Cosine similarity between users: {similarity.item():.4f}")
    
    print("\nInference examples completed!")

if __name__ == "__main__":
    import torch  # Import here for the similarity example
    main() 