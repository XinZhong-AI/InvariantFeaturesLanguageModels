import torch

from utils import covariance, get_last_token_index


@torch.no_grad()
def forward_full(model, tokenizer, text, device):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    ).to(device)

    outputs = model(
        **inputs,
        output_hidden_states=True,
        return_dict=True
    )

    return inputs, outputs.hidden_states, outputs.logits


@torch.no_grad()
def get_feature(
    model,
    tokenizer,
    text,
    hidden_idx,
    device
):

    inputs, hs, _ = forward_full(
        model,
        tokenizer,
        text,
        device
    )

    last_idx = get_last_token_index(inputs)

    z = hs[hidden_idx][0, last_idx, :].float().to(device)

    return z


def generalized_eig(
    S_sp,
    S_sc,
    k=32,
    eps=1e-4
):

    d = S_sp.shape[0]

    I = torch.eye(d, device=S_sp.device)

    S_sp_reg = S_sp + eps * I

    A = torch.linalg.solve(S_sp_reg, S_sc)

    eigvals, eigvecs = torch.linalg.eig(A)

    eigvals = eigvals.real
    eigvecs = eigvecs.real

    idx = torch.argsort(eigvals, descending=True)

    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    U_inv = eigvecs[:, :k]
    U_nuis = eigvecs[:, -k:]

    U_inv, _ = torch.linalg.qr(U_inv)
    U_nuis, _ = torch.linalg.qr(U_nuis)

    return eigvals, U_inv, U_nuis


def decompose(h, U_inv):

    h_inv = U_inv @ (U_inv.T @ h)

    h_nuis = h - h_inv

    return h_inv, h_nuis


@torch.no_grad()
def build_covariance(
    model,
    tokenizer,
    group,
    hidden_idx,
    device
):

    z_orig = get_feature(
        model,
        tokenizer,
        group["original"],
        hidden_idx,
        device
    )

    sp_feats = [
        get_feature(
            model,
            tokenizer,
            t,
            hidden_idx,
            device
        )
        for t in group["sp"]
    ]

    sc_feats = [
        get_feature(
            model,
            tokenizer,
            t,
            hidden_idx,
            device
        )
        for t in group["sc"]
    ]

    delta_sp = torch.stack([
        z - z_orig for z in sp_feats
    ])

    delta_sc = torch.stack([
        z - z_orig for z in sc_feats
    ])

    S_sp = covariance(delta_sp)
    S_sc = covariance(delta_sc)

    return S_sp, S_sc


@torch.no_grad()
def compute_centroid(
    model,
    tokenizer,
    group,
    hidden_idx,
    device
):

    texts = [group["original"]] + group["sp"]

    feats = [
        get_feature(
            model,
            tokenizer,
            t,
            hidden_idx,
            device
        )
        for t in texts
    ]

    return torch.stack(feats).mean(dim=0)