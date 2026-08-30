from __future__ import annotations


def target_only_loss(model, input_ids, attention_mask, labels):
    """Memory-bounded causal loss on target predictor positions only."""
    import torch
    import torch.nn.functional as functional

    target_positions = torch.nonzero(labels[0] != -100, as_tuple=False).squeeze(-1)
    if not target_positions.numel() or int(target_positions.min()) <= 0:
        raise RuntimeError("invalid target-only labels")
    predictor_positions = target_positions - 1
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=None,
        logits_to_keep=predictor_positions,
        use_cache=False,
    )
    loss = functional.cross_entropy(
        output.logits[0].float(), labels[0, target_positions], reduction="mean"
    )
    return loss, int(target_positions.numel())
