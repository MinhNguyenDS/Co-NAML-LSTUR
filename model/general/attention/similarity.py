import torch
import torch.nn.functional as F


class SimilarityAttention(torch.nn.Module):
    """
    A general attention module based on similarity w.r.t. another vector
    """

    def __init__(self):
        super(SimilarityAttention, self).__init__()

    def forward(self, wrt_vector, candidate_vector):
        """
        Args:
            wrt_vector: batch_size, num_candidates, candidate_vector_dim
            candidate_vector: batch_size, candidate_size, candidate_vector_dim
        Returns:
            (shape) batch_size, num_candidates, candidate_vector_dim
        """
        # batch_size, num_candidates, candidate_size
        candidate_weights = F.softmax(
            torch.bmm(wrt_vector, candidate_vector.transpose(1, 2)), dim=2
        )
        # batch_size, num_candidates, candidate_vector_dim
        target = torch.bmm(candidate_weights, candidate_vector)
        return target
