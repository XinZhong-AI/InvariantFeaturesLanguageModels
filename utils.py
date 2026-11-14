import torch
import torch.nn.functional as F


def get_last_token_index(inputs):

    return inputs["attention_mask"].sum(dim=1).item() - 1


def covariance(delta):

    return (delta.T @ delta) / delta.shape[0]


def compare_logits(logits_ref, logits_new):

    ref = logits_ref[:, -1, :].float()

    new = logits_new[:, -1, :].float()

    mse = F.mse_loss(new, ref).item()

    p_ref = F.log_softmax(ref, dim=-1)

    p_new = F.log_softmax(new, dim=-1)

    kl = F.kl_div(
        p_new,
        p_ref.exp(),
        reduction="batchmean"
    ).item()

    return mse, kl