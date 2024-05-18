import math

import torch
import numpy as np

def get_angular_distance_ith_token(i: int) -> float:
    def angular_distance_ith_token(input, output) -> float:
        cos_sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        try:
            input_last_hidden_state = input[:, i, :]
            output_last_hidden_state = output[:, i, :]
        except IndexError as e:
            print(
                f"{e}\nMake sure each sequence in the dataset no shorter than `{i}`-tokens."
            )
            raise RuntimeError

        sim = cos_sim(input_last_hidden_state, output_last_hidden_state)
        sim = torch.clamp(sim, -1.0, 1.0)
        dist = (1 / math.pi) * torch.acos(sim).item()
        return dist

    return angular_distance_ith_token


angular_distance_last_token = get_angular_distance_ith_token(-1)


def angular_distance_all_tokens(input, output):
    cos_sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    sequence_length = input.shape[1]

    sim = cos_sim(input, output)
    sim = torch.clamp(sim, -1.0, 1.0)

    dist = (1 / math.pi) * torch.acos(sim)
    dist = (dist / sequence_length).mean().item()
    return dist

def get_linear_approximation_ith_token(i: int) -> float:
    def linear_approximation(input, output) -> float:
        try:
            input_last_hidden_state = input[:, i, :].numpy()
            output_last_hidden_state = output[:, i, :].numpy()
        except IndexError as e:
            print(
                f"{e}\nMake sure each sequence in the dataset no shorter than `{i}`-tokens."
            )
            raise RuntimeError

        A_est = np.linalg.pinv(input_last_hidden_state.T).dot(output_last_hidden_state.T).T
        output_est = A_est.dot(input_last_hidden_state)
        diff_squared = (output_est - output_last_hidden_state)**2
        diff_summ = diff_squared.sum()
        diff = diff_summ / diff_summ.size
        return np.log(diff)

    return linear_approximation

def get_euclidian_dist_ith_token(i: int) -> float:
    def euclidian_dist_ith_token(input, output) -> float:
        try:
            input_last_hidden_state = input[:, i, :]
            output_last_hidden_state = output[:, i, :]
        except IndexError as e:
            print(
                f"{e}\nMake sure each sequence in the dataset no shorter than `{i}`-tokens."
            )
            raise RuntimeError

        dist = torch.sqrt(torch.sum(torch.pow(torch.subtract(input_last_hidden_state, output_last_hidden_state), 2))) 
        return dist.item()

    return euclidian_dist_ith_token
