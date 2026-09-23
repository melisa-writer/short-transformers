import math
from typing import Callable

import numpy as np
import torch

Distance = Callable[[torch.Tensor, torch.Tensor], float]


def _ith_token(i: int, input, output):
    try:
        return input[:, i, :], output[:, i, :]
    except IndexError as e:
        raise RuntimeError(
            f"Make sure each sequence in the dataset has at least {abs(i) if i < 0 else i + 1} tokens."
        ) from e


# The Unreasonable Ineffectiveness of the Deeper Layers
# https://arxiv.org/abs/2403.17887
# @TODO: make it work with batches
def get_angular_distance_ith_token(i: int) -> Distance:
    def angular_distance_ith_token(input, output) -> float:
        cos_sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        input_last_hidden_state, output_last_hidden_state = _ith_token(i, input, output)

        sim = cos_sim(input_last_hidden_state, output_last_hidden_state)
        sim = torch.clamp(sim, -1.0, 1.0)
        dist = (1 / math.pi) * torch.acos(sim).item()
        return dist

    return angular_distance_ith_token


angular_distance_last_token = get_angular_distance_ith_token(-1)


def angular_distance_all_tokens(input, output):
    cos_sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    sim = cos_sim(input, output)
    sim = torch.clamp(sim, -1.0, 1.0)

    dist = (1 / math.pi) * torch.acos(sim)
    dist = dist.mean().item()
    return dist

# Similar to: Your Transformer is Secretly Linear
# https://arxiv.org/pdf/2405.12250
def get_linear_approximation_ith_token(i: int) -> Distance:
    def linear_approximation_ith_token(input, output) -> float:
        input_last_hidden_state, output_last_hidden_state = _ith_token(i, input, output)
        # numpy has no bfloat16
        input_last_hidden_state = input_last_hidden_state.float().numpy()
        output_last_hidden_state = output_last_hidden_state.float().numpy()

        A_est = np.linalg.pinv(input_last_hidden_state.T).dot(output_last_hidden_state.T).T
        output_est = A_est.dot(input_last_hidden_state)
        diff = ((output_est - output_last_hidden_state) ** 2).mean()
        return float(np.log(diff))

    return linear_approximation_ith_token

# def linear_approximation_all_tokens(input, output):
#     A_est = np.linalg.pinv(input.T).dot(output.T).T
#     output_est = A_est.dot(input)
#     diff_squared = (output_est - output)**2
#     diff_summ = diff_squared.sum()
#     diff = diff_summ / diff_summ.size
#     return diff


def get_euclidian_dist_ith_token(i: int) -> Distance:
    def euclidian_dist_ith_token(input, output) -> float:
        input_last_hidden_state, output_last_hidden_state = _ith_token(i, input, output)
        return torch.dist(input_last_hidden_state, output_last_hidden_state).item()

    return euclidian_dist_ith_token

# @TODO def get_euclidean_dist_all_tokens

# ShortGPT: Layers in Large Language Models are More Redundant Than You Expect
# https://arxiv.org/pdf/2403.03853
def bi_score(input, output) -> float:
    cos_sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    sim = cos_sim(input, output).mean().item()
    return 1 - sim

# Weight subcloning: direct initialization of transformers using larger pretrained ones
# https://arxiv.org/abs/2312.09299
# ||f(x)|| / ||x + f(x)|| where x is the block input and x + f(x) its output
def relative_magnitude(input, output, eps=1e-6) -> float:
    score = torch.norm(output - input, dim=-1) / (torch.norm(output, dim=-1) + eps)
    return score.mean().item()
