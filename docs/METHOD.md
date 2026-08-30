# COGA method

For a success trajectory and failure trajectory in the same family/prompt/policy cohort,
COGA computes target-only LoRA gradients on the final assistant target:

```text
g+ = grad_phi L_target(success)
g- = grad_phi L_target(failure)
d  = CountSketch(g+ - g-)
```

The 4-bit Qwen3-8B base is frozen; only q/k/v/o LoRA parameters are trainable. A 128-step
success warm-up avoids measuring only the degenerate zero-initialization slice. Gradient
extraction runs in evaluation mode and stores an 8,192-dimensional CountSketch rather than a
dense gradient.

Family/prompt components are assigned to five deterministic folds. A row is aligned only
against the normalized mean of real contrasts outside its fold. All controls use that same
real prototype:

- reward shuffle;
- turn shuffle;
- cross-cohort failure pairing;
- length-matched cross-task swap;
- bag of actions;
- random labels;
- self-pair identity.

The frozen gate is the paired cohort bootstrap 95% confidence interval of
`real - strongest control`; its lower bound must be greater than zero. A failed gate prevents
construction of a result-selected SFT arm.

After a pass, the top 50% of real cohorts form the COGA arm. A deterministic equal-cohort
rejection-success sample forms the baseline. Both arms consume exactly 1,000,000 target loss
tokens; the final sample's labels are capped rather than exceeding the budget.

Both adapters start independently from the same pinned base revision and use one epoch,
batch size one, no packing, NF4/BF16 QLoRA, and target-only causal loss. Terminal-Bench tests
long-horizon environment interaction; BFCL V4 Multi-Turn tests multi-step function/tool
calling. Neither benchmark is used to tune selection.
