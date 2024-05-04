import math

import torch


def angular_distance_last_token(input, output):
    cos_sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    input_last_hidden_state = input[:, -1, :]
    output_last_hidden_state = output[:, -1, :]

    sim = cos_sim(input_last_hidden_state, output_last_hidden_state)
    dist = (1 / math.pi) * torch.acos(sim).item()
    return dist


def angular_distance_all_tokens(input, output):
    cos_sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    sequence_length = input.shape[1]

    sim = cos_sim(input, output)
    sim = torch.clamp(sim, -1.0, 1.0)

    # @TODO sometimes there is a nan here, inspect why
    dist = (1 / math.pi) * torch.acos(sim)
    dist = (dist / sequence_length).mean().item()
    return dist
