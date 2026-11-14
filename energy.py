import torch
import pandas as pd
import os

from subspace import (
    build_covariance,
    generalized_eig,
    compute_centroid
)


def compute_E_nuisance(delta, U_nuis):

    P_nuis = U_nuis @ U_nuis.T

    proj = P_nuis @ delta

    E = torch.norm(proj) ** 2 / (
        torch.norm(delta) ** 2 + 1e-8
    )

    return E.item()


@torch.no_grad()
def run_energy(
    model,
    tokenizer,
    device,
    groups,
    k,
    eps,
    output_path
):

    rows = []

    num_layers = len(model.model.layers) + 1

    for g_id, g in enumerate(groups):

        for h_id, h in enumerate(groups):

            if g_id == h_id:
                continue

            for layer in range(num_layers):

                print(
                    f"g={g_id}, h={h_id}, layer={layer}"
                )

                S_sp, S_sc = build_covariance(
                    model,
                    tokenizer,
                    g,
                    layer,
                    device
                )

                eigvals, U_inv, U_nuis = generalized_eig(
                    S_sp,
                    S_sc,
                    k=k,
                    eps=eps
                )

                z_g = compute_centroid(
                    model,
                    tokenizer,
                    g,
                    layer,
                    device
                )

                z_h = compute_centroid(
                    model,
                    tokenizer,
                    h,
                    layer,
                    device
                )

                delta_gh = z_g - z_h

                E_nuis = compute_E_nuisance(
                    delta_gh,
                    U_nuis
                )

                rows.append({
                    "group_g": g_id,
                    "group_h": h_id,
                    "layer": layer,

                    "top_lambda_g": eigvals[0].item(),

                    "E_nuisance": E_nuis,

                })

    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(df.head())