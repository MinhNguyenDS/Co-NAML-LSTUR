import torch
import torch.nn as nn
import torch.nn.functional as F
from model.NAML_LSTUR.news_encoder import NewsEncoder
from model.NAML_LSTUR.user_encoder import UserEncoder

# from model.general.click_predictor.DNN import DNNClickPredictor
from model.general.click_predictor.dot_product import DotProductClickPredictor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class NAML_LSTUR(torch.nn.Module):
    """nn
    NAML_LSTUR network.
    Input 1 + K candidate news and a list of user clicked news, produce the click probability.
    """

    def __init__(self, config, pretrained_word_embedding=None):
        super(NAML_LSTUR, self).__init__()
        self.config = config
        self.news_encoder = NewsEncoder(config, pretrained_word_embedding)
        self.user_encoder = UserEncoder(config)

        self.click_predictor = DotProductClickPredictor()
        # self.click_predictor = NAML_DNNClickPredictor(int(self.config.num_filters / self.config.window_size) * (self.config.window_size + 1))

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
        sizeC = candidate_news_vector.size()
        sizeU = user_vector.size()
        value = int(self.config.num_filters / 100)
        # click_probability = self.click_predictor(candidate_news_vector.view(sizeC[0] * sizeC[1], sizeC[2]),
        #                                          user_vector.view(sizeU[0] * value, int(sizeU[1] / value))).view(sizeC[0], sizeC[1])
        # click_probability = self.click_predictor(candidate_news_vector.view(384, 300),
        #                                          user_vector.view(384, 100)).view(128, 3)
        click_probability = self.click_predictor(candidate_news_vector, user_vector)
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
        # candidate_size
        return self.click_predictor(
            news_vector.unsqueeze(dim=0), user_vector.unsqueeze(dim=0)
        ).squeeze(dim=0)

        # return self.click_predictor(
        #     news_vector.unsqueeze(dim=0).view(384, 300),
        #     user_vector.unsqueeze(dim=0).view(384, 100)).view(128, 3).squeeze(dim=0)
