import torch
import pandas as pd
import os

from subspace import (
    forward_full,
    generalized_eig,
    decompose
)

from utils import (
    compare_logits,
    get_last_token_index
)


@torch.no_grad()
def run_with_layer_patch(
    model,
    tokenizer,
    original_text,
    hidden_idx,
    patched_last_hidden,
    device
):

    inputs = tokenizer(
        original_text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    ).to(device)

    last_idx = get_last_token_index(inputs)

    if hidden_idx == 0:

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        embeds = model.model.embed_tokens(input_ids)

        embeds = embeds.clone()

        embeds[0, last_idx, :] = patched_last_hidden.to(
            device=embeds.device,
            dtype=embeds.dtype
        )

        outputs = model(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            return_dict=True
        )

        return outputs.logits

    layer_id = hidden_idx - 1

    def hook_fn(module, input, output):

        hidden = output[0]

        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)

        hidden = hidden.clone()

        hidden[0, last_idx, :] = patched_last_hidden.to(
            device=hidden.device,
            dtype=hidden.dtype
        )

        return hidden

    handle = model.model.layers[layer_id].register_forward_hook(hook_fn)

    outputs = model(
        **inputs,
        return_dict=True
    )

    handle.remove()

    return outputs.logits


@torch.no_grad()
def run_intervention(
    model,
    tokenizer,
    device,
    groups,
    k,
    alpha,
    eps,
    output_path
):

    all_rows = []

    num_hidden_states = len(model.model.layers) + 1

    for group_id, group in enumerate(groups):

        inputs_orig, hs_orig, logits_orig = forward_full(
            model,
            tokenizer,
            group["original"],
            device
        )

        orig_last = get_last_token_index(inputs_orig)

        for hidden_idx in range(num_hidden_states):

            print(f"group={group_id}, layer={hidden_idx}")

            z_orig = hs_orig[hidden_idx][0, orig_last, :].float()

            sp_features = []

            for text in group["sp"]:

                inputs_sp, hs_sp, _ = forward_full(
                    model,
                    tokenizer,
                    text,
                    device
                )

                sp_last = get_last_token_index(inputs_sp)

                z_sp = hs_sp[hidden_idx][0, sp_last, :].float()

                sp_features.append(z_sp)

            sc_features = []

            for text in group["sc"]:

                inputs_sc, hs_sc, _ = forward_full(
                    model,
                    tokenizer,
                    text,
                    device
                )

                sc_last = get_last_token_index(inputs_sc)

                z_sc = hs_sc[hidden_idx][0, sc_last, :].float()

                sc_features.append(z_sc)

            delta_sp = torch.stack([
                z - z_orig for z in sp_features
            ])

            delta_sc = torch.stack([
                z - z_orig for z in sc_features
            ])

            S_sp = (delta_sp.T @ delta_sp) / delta_sp.shape[0]

            S_sc = (delta_sc.T @ delta_sc) / delta_sc.shape[0]

            eigvals, U_inv, U_nuis = generalized_eig(
                S_sp,
                S_sc,
                k=k,
                eps=eps
            )

            h_orig_inv, h_orig_nuis = decompose(
                z_orig,
                U_inv
            )

            sp_nuis_list = []

            for z_sp in sp_features:

                _, h_sp_nuis = decompose(
                    z_sp,
                    U_inv
                )

                sp_nuis_list.append(h_sp_nuis)

            h_sp_nuis_avg = torch.stack(
                sp_nuis_list
            ).mean(dim=0)

            h_nuis_patch = z_orig + alpha * (
                h_sp_nuis_avg - h_orig_nuis
            )

            logits_nuis = run_with_layer_patch(
                model,
                tokenizer,
                group["original"],
                hidden_idx,
                h_nuis_patch,
                device
            )

            nuis_mse, nuis_kl = compare_logits(
                logits_orig,
                logits_nuis
            )

            all_rows.append({
                "group_id": group_id,
                "layer": hidden_idx,
                "top_lambda": eigvals[0].item(),

                "nuis_mse": nuis_mse,
                "nuis_kl": nuis_kl,
            })

    df = pd.DataFrame(all_rows)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(df.head())