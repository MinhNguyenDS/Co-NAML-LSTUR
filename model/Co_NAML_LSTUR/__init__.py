import torch
import torch.nn as nn
import torch.nn.functional as F


from model.Co_NAML_LSTUR.NAML.news_encoder import NewsEncoder
from model.Co_NAML_LSTUR.LSTUR.user_encoder import UserEncoder
from model.Co_NAML_LSTUR.click_predictor.DNN import DNNClickPredictor
from model.Co_NAML_LSTUR.DKN.attention import Attention
# from model.general.click_predictor.dot_product import DotProductClickPredictor

# Updated device selection to prefer MPS on Mac
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


class Co_NAML_LSTUR(torch.nn.Module):
    """
    Co_NAML_LSTUR network.
    Input 1 + K candidate news and a list of user clicked news, produce the click probability.
    """

    def __init__(self, config, pretrained_word_embedding=None):
        super(Co_NAML_LSTUR, self).__init__()
        self.config = config
        self.news_encoder = NewsEncoder(config, pretrained_word_embedding)
        self.user_encoder = UserEncoder(config)

        # self.click_predictor = DotProductClickPredictor()
        self.click_predictor = DNNClickPredictor(
            input_size=self.config.num_filters
            + self.config.num_filters,  # news vector dim + user vector dim
            hidden_size=128,  # Specify hidden size explicitly
        )

        self.attention = Attention(config)

        assert int(config.num_filters * 1.5) == config.num_filters * 1.5
        self.user_embedding = nn.Embedding(
            config.num_users, int(config.num_filters * 1.5), padding_idx=0
        )

    def forward(self, user, clicked_news_length, candidate_news, clicked_news):
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

        # LSTUR
        # con: batch_size, num_filters * 1.5
        # TODO what if not drop
        user = F.dropout1d(
            self.user_embedding(user.to(device)).unsqueeze(dim=0),
            p=self.config.masking_probability,
            training=self.training,
        ).squeeze(dim=0)

        # batch_size, num_filters
        user_vector = self.user_encoder(user, clicked_news_length, clicked_news_vector)
        # batch_size, 1 + K

        # Get actual dimensions from the tensors
        batch_size, num_candidates, news_vector_dim = candidate_news_vector.size()
        _, user_vector_dim = user_vector.size()

        # Use actual dimensions instead of hardcoded values
        click_probability = self.click_predictor(
            candidate_news_vector.view(batch_size * num_candidates, news_vector_dim),
            user_vector.unsqueeze(1)
            .expand(-1, num_candidates, -1)
            .reshape(batch_size * num_candidates, user_vector_dim),
        ).view(batch_size, num_candidates)

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

    def get_user_vector(self, user, clicked_news_length, clicked_news_vector):
        """
        Args:
            user: batch_size
            clicked_news_length: batch_size
            clicked_news_vector: batch_size, num_clicked_news_a_user, num_filters * 3
        Returns:
            (shape) batch_size, num_filters * 3
        """

        user = self.user_embedding(user.to(device))
        # batch_size, num_filters * 3
        return self.user_encoder(user, clicked_news_length, clicked_news_vector)

    def get_prediction(self, news_vector, user_vector):
        """
        Args:
            news_vector: candidate_size, word_embedding_dim
            user_vector: word_embedding_dim
        Returns:
            click_probability: candidate_size
        """
        # Add batch dimension to user_vector if not present
        user_vector = (
            user_vector.unsqueeze(0) if user_vector.dim() == 1 else user_vector
        )

        # Apply attention mechanism to get user-aware news representation
        user_vector_expanded = user_vector.expand(news_vector.size(0), -1)
        if hasattr(self, "attention"):
            # If model has attention mechanism
            user_vector_expanded = self.attention(
                news_vector,
                user_vector.unsqueeze(0).expand(news_vector.size(0), -1, -1),
            )

        # Use click predictor to get final prediction
        click_probability = self.click_predictor(news_vector, user_vector_expanded)
        return click_probability
