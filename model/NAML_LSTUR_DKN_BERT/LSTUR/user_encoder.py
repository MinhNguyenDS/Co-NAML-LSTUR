import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class UserEncoder(torch.nn.Module):
    def __init__(self, config):
        super(UserEncoder, self).__init__()
        self.config = config
        self.lstm = nn.LSTM(
            config.num_filters, int(config.num_filters * 1.5), batch_first=True
        )
        self.linear = nn.Linear(int(config.num_filters * 3), config.num_filters)

    def forward(self, user, clicked_news_length, clicked_news_vector):
        """
        Args:
            user:
                ini: batch_size, num_filters * 3
                con: batch_size, num_filters * 1.5
            clicked_news_length: batch_size,
            clicked_news_vector: batch_size, num_clicked_news_a_user, num_filters * 3
        Returns:
            (shape) batch_size, num_filters * 3
        """
        clicked_news_length[clicked_news_length == 0] = 1
        # 1, batch_size, num_filters * 3
        packed_clicked_news_vector = pack_padded_sequence(
            clicked_news_vector,
            clicked_news_length.cpu(),  # lengths must be on CPU for pack_padded_sequence
            batch_first=True,
            enforce_sorted=False,
        )

        # LSTM returns output, (last_hidden, last_cell)
        # We only need the last hidden state
        _, (last_hidden, _) = self.lstm(packed_clicked_news_vector)

        news_encoder = torch.cat((last_hidden.squeeze(dim=0), user), dim=1)
        news_encoder = self.linear(news_encoder)
        return news_encoder
