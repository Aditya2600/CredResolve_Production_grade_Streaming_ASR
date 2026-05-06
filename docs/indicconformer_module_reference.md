# IndicConformer Module Reference

This document records the useful parts of a NeMo module-name inspection for:

```text
/home/ubuntu/models/indicconformer/IndicConformer.nemo
```

The inspected model restores as `EncDecHybridRNNTCTCBPEModel`. The current
Vaani PEFT path uses NeMo encoder adapters, not full fine-tuning. Use this
reference when checking where adapters are inserted, when choosing module-name
patterns for inspection, or when comparing a future model export against the
known IndicConformer structure.

## Inspect Command

Run this from the repo root in an environment with NeMo ASR installed:

```bash
python tools/run_nemo_adapter_peft.py \
  --model /home/ubuntu/models/indicconformer/IndicConformer.nemo \
  --inspect-module-names
```

The default inspection patterns are:

```text
q,k,v,proj,linear,ffn
```

Override them when you need a wider or narrower inventory:

```bash
python tools/run_nemo_adapter_peft.py \
  --model /home/ubuntu/models/indicconformer/IndicConformer.nemo \
  --inspect-module-names \
  --inspect-module-name-patterns q,k,v,proj,linear,ffn,conv,norm
```

The inspect mode temporarily swaps the encoder config to the adapter-compatible
Conformer encoder class, restores the model, prints matching module names, and
exits before any training data is required.

## Log Noise

The raw inspection log can include NeMo restore messages, tokenizer setup logs,
and PyTorch deprecation warnings from Megatron tensor-parallel layers. These are
not adapter target names. The useful output starts after the successful model
restore line and consists of entries like:

```text
encoder.layers.0.self_attn.linear_q torch.nn.modules.linear.Linear
```

## Top-Level Structure

The observed model has these major areas:

| Prefix | Role | Notes |
| --- | --- | --- |
| `encoder.pre_encode.conv` | Audio subsampling frontend | Sequential Conv2d/ReLU stack before Conformer layers. |
| `encoder.layers.<N>` | Repeated Conformer encoder blocks | Main adapter-relevant encoder body. |
| `encoder.layers.<N>.feed_forward1` | First feed-forward module | Contains `linear1`, activation, and `linear2`. |
| `encoder.layers.<N>.conv` | Conformer convolution module | Contains pointwise, depthwise, batch norm, activation, and output pointwise conv. |
| `encoder.layers.<N>.self_attn` | Self-attention module | Contains query/key/value/output/position linear projections. |
| `encoder.layers.<N>.feed_forward2` | Second feed-forward module | Contains `linear1` and `linear2`. |

The previous raw layer dump captured layers `0` through `14` and ended
mid-line, so treat that file as an incomplete scratch paste rather than the
complete architecture. Rerun the inspect command when you need a complete
current inventory.

## Repeated Encoder Block Pattern

Each captured Conformer block follows this pattern:

```text
encoder.layers.<N>.feed_forward1.linear1       Linear
encoder.layers.<N>.feed_forward1.activation    Swish
encoder.layers.<N>.feed_forward1.linear2       Linear
encoder.layers.<N>.norm_conv                   LayerNorm
encoder.layers.<N>.conv                        ConformerConvolution
encoder.layers.<N>.conv.pointwise_conv1        Conv1d
encoder.layers.<N>.conv.depthwise_conv         CausalConv1D
encoder.layers.<N>.conv.batch_norm             BatchNorm1d
encoder.layers.<N>.conv.activation             Swish
encoder.layers.<N>.conv.pointwise_conv2        Conv1d
encoder.layers.<N>.self_attn.linear_q          Linear
encoder.layers.<N>.self_attn.linear_k          Linear
encoder.layers.<N>.self_attn.linear_v          Linear
encoder.layers.<N>.self_attn.linear_out        Linear
encoder.layers.<N>.self_attn.linear_pos        Linear
encoder.layers.<N>.feed_forward2.linear1       Linear
encoder.layers.<N>.feed_forward2.linear2       Linear
```

This repeated pattern is the most important part of the inspection output. If a
future IndicConformer export changes these names, inspect mode should be rerun
before adapter training or any LoRA-style target selection.

## PEFT Guidance

The supported training path in this repo is adapter-only:

```bash
bash scripts/run_vaani_adapter_peft.sh
```

`tools/run_nemo_adapter_peft.py` freezes the restored model, adds one encoder
adapter, enables only that adapter, and exits if any non-adapter parameter is
trainable. That means the module names above are reference points, not a request
to unfreeze base encoder layers.

Use these names for:

- confirming the restored model has the expected Conformer encoder structure;
- deciding which names to include in `--inspect-module-name-patterns`;
- debugging why an adapter-compatible restore or module inspection changed.

Do not use this list as permission to train `encoder.layers.<N>.*` directly in
the Vaani adapter path. Full fine-tuning and partial encoder unfreezing are
separate workflows with a larger forgetting risk.

## Suggested Artifacts

When preserving a full raw inspection, write it outside `docs/` so the docs stay
readable:

```bash
python tools/run_nemo_adapter_peft.py \
  --model /home/ubuntu/models/indicconformer/IndicConformer.nemo \
  --inspect-module-names \
  > artifacts/indicconformer_module_names.txt
```

Then summarize only the stable patterns here.
