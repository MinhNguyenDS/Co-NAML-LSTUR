import torch
from model.NAML_DKN.news_encoder import NewsEncoder
from model.NAML_DKN.user_encoder import UserEncoder
from model.NAML_DKN.click_predictor.DNN import DNNClickPredictor


class NAML_DKN(torch.nn.Module):
    """
    NAML_DKN network.
    Input 1 + K candidate news and a list of user clicked news, produce the click probability.
    """

    def __init__(self, config, pretrained_word_embedding=None):
        super(NAML_DKN, self).__init__()
        self.config = config
        self.news_encoder = NewsEncoder(config, pretrained_word_embedding)
        self.user_encoder = UserEncoder(config)
        self.click_predictor = DNNClickPredictor(2 * self.config.num_filters)

    def forward(self, candidate_news, clicked_news):
        """
        Args:
            candidate_news:
                [
                    {
                        "category": batch_size,
                        "subcategory": batch_size,
                        "title": batch_size * num_words_title,
                        "abstract": batch_size * num_words_abstract
                    } * (1 + K)
                ]
            clicked_news:
                [
                    {
                        "category": batch_size,
                        "subcategory": batch_size,
                        "title": batch_size * num_words_title,
                        "abstract": batch_size * num_words_abstract
                    } * num_clicked_news_a_user
                ]
        Returns:
            click_probability: batch_size
        """
        # batch_size, 1 + K, num_filters
        candidate_news_vector = torch.stack(
            [self.news_encoder(x) for x in candidate_news], dim=1
        )
        # batch_size, num_clicked_news_a_user, num_filters
        clicked_news_vector = torch.stack(
            [self.news_encoder(x) for x in clicked_news], dim=1
        )
        # batch_size, num_filters
        user_vector = self.user_encoder(clicked_news_vector)

        # ---Click Prediction---
        # Get dimensions for reshape operations
        batch_size = candidate_news_vector.size(0)
        num_candidates = candidate_news_vector.size(1)
        embedding_dim = candidate_news_vector.size(2)

        # Reshape candidate_news_vector to [batch_size * num_candidates, embedding_dim]
        candidate_news_vector_flat = candidate_news_vector.view(-1, embedding_dim)

        # Expand user_vector to match the candidate shape
        # [batch_size, embedding_dim] -> [batch_size, 1, embedding_dim] -> [batch_size, num_candidates, embedding_dim]
        expanded_user_vector = user_vector.unsqueeze(1).expand(-1, num_candidates, -1)
        # Reshape to [batch_size * num_candidates, embedding_dim]
        expanded_user_vector_flat = expanded_user_vector.contiguous().view(
            -1, embedding_dim
        )

        # Calculate click probability using the flattened vectors
        click_probability_flat = self.click_predictor(
            candidate_news_vector_flat, expanded_user_vector_flat
        )

        # Reshape back to [batch_size, num_candidates]
        click_probability = click_probability_flat.view(batch_size, num_candidates)

        return click_probability

    def get_news_vector(self, news):
        """
        Args:
            news:
                {
                    "category": batch_size,
                    "subcategory": batch_size,
                    "title": batch_size * num_words_title,
                    "abstract": batch_size * num_words_abstract
                }
        Returns:
            (shape) batch_size, num_filters
        """
        # batch_size, num_filters
        return self.news_encoder(news)

    def get_user_vector(self, clicked_news_vector):
        """
        Args:
            clicked_news_vector: batch_size, num_clicked_news_a_user, num_filters
        Returns:
            (shape) batch_size, num_filters
        """
        # batch_size, num_filters
        return self.user_encoder(clicked_news_vector)

    def get_prediction(self, candidate_news_vector, user_vector):
        """
        Args:
            candidate_news_vector: [candidate_size, news_embedding_dim]
            user_vector: [user_embedding_dim]
        Returns:
            click_probability: [candidate_size]
        """
        # Repeat user_vector to match candidate_news_vector shape
        candidate_size = candidate_news_vector.size(0)
        user_vector_repeated = user_vector.unsqueeze(0).expand(
            candidate_size, -1
        )  # Shape: [candidate_size, user_embedding_dim]

        # Both tensors are now [candidate_size, embedding_dim]
        return self.click_predictor(candidate_news_vector, user_vector_repeated)
