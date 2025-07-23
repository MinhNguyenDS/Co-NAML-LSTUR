import argparse
import numpy as np
import pandas as pd
from ast import literal_eval
import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
from pathlib import Path
from tqdm import tqdm
import importlib

# Import configuration and model
from config import model_name as default_model_name

# Updated device selection to prefer MPS on Mac
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


class NewsInferenceDataset(Dataset):
    """Dataset for loading news articles for inference."""
    
    def __init__(self, news_path, config):
        super(NewsInferenceDataset, self).__init__()
        self.config = config
        
        # Load news data
        self.news_parsed = pd.read_table(
            news_path,
            usecols=["id"] + config.dataset_attributes["news"],
            converters={
                attribute: literal_eval
                for attribute in set(config.dataset_attributes["news"])
                & set(["title", "abstract", "title_entities", "abstract_entities"])
            },
        )
        
        # Convert to dictionary format for easy access
        self.news2dict = self.news_parsed.to_dict("index")
        for key1 in self.news2dict.keys():
            for key2 in self.news2dict[key1].keys():
                if type(self.news2dict[key1][key2]) != str:
                    self.news2dict[key1][key2] = torch.tensor(
                        self.news2dict[key1][key2]
                    )

    def __len__(self):
        return len(self.news_parsed)

    def __getitem__(self, idx):
        return self.news2dict[idx]
    
    def get_news_by_id(self, news_id):
        """Get news data by news ID."""
        if news_id in self.news2dict:
            return self.news2dict[news_id]
        else:
            # Return padding if news not found
            return self._get_padding()
    
    def _get_padding(self):
        """Get padding tensor for missing news."""
        padding_all = {
            'id': 'PADDED_NEWS',
            'category': torch.tensor(0),
            'subcategory': torch.tensor(0),
            'title': torch.tensor([0] * self.config.num_words_title),
            'abstract': torch.tensor([0] * self.config.num_words_abstract),
        }
        return {k: v for k, v in padding_all.items() 
                if k in ['id'] + self.config.dataset_attributes['news']}


class NewsRecommendationInference:
    """Main inference class for news recommendation."""
    
    def __init__(self, model_name="Co_NAML_LSTUR", checkpoint_path=None):
        self.model_name = model_name
        self.device = device
        
        # Load model configuration
        try:
            Model = getattr(importlib.import_module(f"model.{model_name}"), model_name)
            self.config = getattr(importlib.import_module("config"), f"{model_name}Config")
        except AttributeError:
            raise ValueError(f"{model_name} not included!")
        
        # Initialize and load model
        self.model = Model(self.config).to(self.device)
        self._load_checkpoint(checkpoint_path)
        
        # News dataset for inference
        self.news_dataset = None
        self.news2vector = {}
        
        print(f"Inference system initialized using device: {self.device}")
    
    def _load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        if checkpoint_path is None:
            # Find latest checkpoint
            checkpoint_dir = os.path.join("./checkpoint", self.model_name)
            if not os.path.exists(checkpoint_dir):
                raise FileNotFoundError(f"No checkpoint directory found: {checkpoint_dir}")
            
            all_checkpoints = {
                int(x.split(".")[-2].split("-")[-1]): x 
                for x in os.listdir(checkpoint_dir) if x.endswith('.pth')
            }
            if not all_checkpoints:
                raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
            
            checkpoint_path = os.path.join(checkpoint_dir, all_checkpoints[max(all_checkpoints.keys())])
        
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        print("Model loaded successfully!")
    
    def load_news_data(self, news_path="./data/test/news_parsed.tsv"):
        """Load news dataset for inference."""
        print("Loading news data...")
        self.news_dataset = NewsInferenceDataset(news_path, self.config)
        print(f"Loaded {len(self.news_dataset)} news articles")
    
    def precompute_news_vectors(self, batch_size=None):
        """Precompute vectors for all news articles."""
        if self.news_dataset is None:
            raise ValueError("News data not loaded. Call load_news_data() first.")
        
        if batch_size is None:
            batch_size = max(1, self.config.batch_size)
        
        print("Precomputing news vectors...")
        news_dataloader = DataLoader(
            self.news_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,  # Reduce for MPS compatibility
            drop_last=False,
            pin_memory=False,  # Disable for MPS
        )
        
        self.news2vector = {}
        with torch.no_grad():
            for minibatch in tqdm(news_dataloader, desc="Computing news vectors"):
                news_ids = minibatch["id"]
                # Remove 'id' from the batch for model input
                model_input = {k: v for k, v in minibatch.items() if k != "id"}
                
                news_vector = self.model.get_news_vector(model_input)
                for news_id, vector in zip(news_ids, news_vector):
                    if news_id not in self.news2vector:
                        self.news2vector[news_id] = vector.cpu()  # Store on CPU to save memory
        
        # Add padding vector
        if self.news2vector:
            sample_vector = list(self.news2vector.values())[0]
            self.news2vector["PADDED_NEWS"] = torch.zeros_like(sample_vector)
        
        print(f"Precomputed vectors for {len(self.news2vector)} news articles")
    
    def get_user_vector(self, user_id, clicked_news_ids):
        """Get user vector based on clicked news history."""
        if not self.news2vector:
            raise ValueError("News vectors not computed. Call precompute_news_vectors() first.")
        
        # Limit clicked news to model's maximum
        clicked_news_ids = clicked_news_ids[-self.config.num_clicked_news_a_user:]
        clicked_news_length = len(clicked_news_ids)
        
        # Pad with PADDED_NEWS if necessary
        padding_needed = self.config.num_clicked_news_a_user - len(clicked_news_ids)
        clicked_news_ids = ["PADDED_NEWS"] * padding_needed + clicked_news_ids
        
        # Get news vectors for clicked news
        clicked_news_vectors = []
        for news_id in clicked_news_ids:
            if news_id in self.news2vector:
                clicked_news_vectors.append(self.news2vector[news_id].to(self.device))
            else:
                clicked_news_vectors.append(self.news2vector["PADDED_NEWS"].to(self.device))
        
        # Stack vectors: [num_clicked_news_a_user, num_filters]
        clicked_news_vector = torch.stack(clicked_news_vectors, dim=0).unsqueeze(0)  # Add batch dim
        
        # Prepare user inputs
        user_tensor = torch.tensor([user_id], dtype=torch.long)
        clicked_news_length_tensor = torch.tensor([clicked_news_length], dtype=torch.long)
        
        with torch.no_grad():
            user_vector = self.model.get_user_vector(
                user_tensor, 
                clicked_news_length_tensor, 
                clicked_news_vector
            )
        
        return user_vector.squeeze(0)  # Remove batch dimension
    
    def predict(self, user_id, clicked_news_ids, candidate_news_ids):
        """Make predictions for candidate news given user's history."""
        # Get user vector
        user_vector = self.get_user_vector(user_id, clicked_news_ids)
        
        # Get candidate news vectors
        candidate_vectors = []
        for news_id in candidate_news_ids:
            if news_id in self.news2vector:
                candidate_vectors.append(self.news2vector[news_id].to(self.device))
            else:
                candidate_vectors.append(self.news2vector["PADDED_NEWS"].to(self.device))
        
        candidate_news_vector = torch.stack(candidate_vectors, dim=0)
        
        with torch.no_grad():
            click_probabilities = self.model.get_prediction(candidate_news_vector, user_vector)
        
        # Convert to numpy and pair with news IDs
        probs = click_probabilities.cpu().numpy()
        results = list(zip(candidate_news_ids, probs))
        
        # Sort by probability (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def recommend_top_k(self, user_id, clicked_news_ids, candidate_news_ids, k=10):
        """Get top-k news recommendations."""
        predictions = self.predict(user_id, clicked_news_ids, candidate_news_ids)
        return predictions[:k]
    
    def batch_predict(self, user_data_list, candidate_news_ids):
        """Batch prediction for multiple users."""
        results = {}
        
        for user_data in tqdm(user_data_list, desc="Processing users"):
            user_id = user_data['user_id']
            clicked_news_ids = user_data['clicked_news_ids']
            
            try:
                predictions = self.predict(user_id, clicked_news_ids, candidate_news_ids)
                results[user_id] = predictions
            except Exception as e:
                print(f"Error processing user {user_id}: {e}")
                results[user_id] = []
        
        return results


def parse_args():
    parser = argparse.ArgumentParser(description="News recommendation inference")
    parser.add_argument("--model_name", type=str, default="Co_NAML_LSTUR",
                       help="Name of the model")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                       help="Path to model checkpoint")
    parser.add_argument("--news_data", type=str, default="./data/test/news_parsed.tsv",
                       help="Path to news data file")
    parser.add_argument("--user_id", type=int, default=1,
                       help="User ID for single prediction")
    parser.add_argument("--clicked_news", type=str, nargs="+", 
                       default=["N37378", "N14827", "N50398"],
                       help="List of clicked news IDs")
    parser.add_argument("--candidate_news", type=str, nargs="+",
                       default=["N37378", "N14827", "N50398", "N48265", "N42793"],
                       help="List of candidate news IDs")
    parser.add_argument("--top_k", type=int, default=5,
                       help="Number of top recommendations to return")
    parser.add_argument("--batch_file", type=str, default=None,
                       help="JSON file with batch prediction data")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize inference system
    inferencer = NewsRecommendationInference(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path
    )
    
    # Load news data and precompute vectors
    inferencer.load_news_data(args.news_data)
    inferencer.precompute_news_vectors()
    
    if args.batch_file:
        # Batch prediction mode
        print(f"Loading batch data from {args.batch_file}")
        with open(args.batch_file, 'r') as f:
            batch_data = json.load(f)
        
        user_data_list = batch_data['users']
        candidate_news_ids = batch_data['candidate_news']
        
        results = inferencer.batch_predict(user_data_list, candidate_news_ids)
        
        # Save results
        output_file = args.batch_file.replace('.json', '_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Batch results saved to {output_file}")
    
    else:
        # Single prediction mode
        print(f"\nMaking prediction for user {args.user_id}")
        print(f"Clicked news: {args.clicked_news}")
        print(f"Candidate news: {args.candidate_news}")
        
        try:
            recommendations = inferencer.recommend_top_k(
                user_id=args.user_id,
                clicked_news_ids=args.clicked_news,
                candidate_news_ids=args.candidate_news,
                k=args.top_k
            )
            
            print(f"\nTop {args.top_k} recommendations:")
            for i, (news_id, prob) in enumerate(recommendations, 1):
                print(f"{i}. News {news_id}: {prob:.4f}")
                
        except Exception as e:
            print(f"Error during prediction: {e}")


if __name__ == "__main__":
    main() 