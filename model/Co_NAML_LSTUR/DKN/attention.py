import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(torch.nn.Module):
    """
    Attention Net.
    Input embedding vectors (produced by KCNN) of a candidate news and all of user's clicked news,
    produce final user embedding vectors with respect to the candidate news.
    """

    def __init__(self, config):
        super(Attention, self).__init__()
        self.config = config
        # Calculate the dimensions correctly from config
        self.news_dim = config.num_filters
        self.concat_dim = self.news_dim * 2  # news vector + user vector

        self.dnn = nn.Sequential(nn.Linear(self.concat_dim, 16), nn.Linear(16, 1))

    def forward(self, candidate_news_vector, clicked_news_vector):
        """
        Args:
            candidate_news_vector: batch_size, len(window_sizes) * num_filters
            clicked_news_vector: batch_size, num_clicked_news_a_user, len(window_sizes) * num_filters
        Returns:
            user_vector: batch_size, len(window_sizes) * num_filters
        """
        # Ensure dimensions are correct
        batch_size = clicked_news_vector.size(0)
        num_clicked = clicked_news_vector.size(1)

        # Expand candidate news to match clicked news for concatenation
        expanded_candidate = candidate_news_vector.unsqueeze(1).expand(
            -1, num_clicked, -1
        )

        # Concatenate along feature dimension
        concat_vectors = torch.cat((expanded_candidate, clicked_news_vector), dim=2)

        # Calculate attention weights
        attention_scores = self.dnn(concat_vectors).squeeze(dim=2)
        clicked_news_weights = F.softmax(attention_scores, dim=1)

        # Apply attention weights to get user vector
        user_vector = torch.bmm(
            clicked_news_weights.unsqueeze(dim=1), clicked_news_vector
        ).squeeze(dim=1)

        return user_vector
