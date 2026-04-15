# TensorBoard Guide For NeMo ASR Fine-Tuning

This guide is for the NeMo fine-tuning runs written under `artifacts/ft_runs/...`.

## Recommended Layout

Apply the repo's grouped TensorBoard layout to a run directory:

```bash
conda activate nemo_asr
cd /home/ubuntu/CredResolve_Production_grade_Streaming_ASR

python tools/apply_tensorboard_layout.py \
  artifacts/ft_runs/prod_same_model_full/indicconformer_prod_mined_ft_full/2026-04-08_13-23-58 \
  --list-tags
```

Then launch TensorBoard:

```bash
tensorboard --logdir artifacts/ft_runs --bind_all
```

The layout groups the Scalars tab into:

- `01 Model Quality`
- `02 Losses`
- `03 Optimization`
- `04 Speed`

## What Each Graph Means

### Validation WER

Tags:

- `val_wer`
- `val_wer_ctc`

Purpose:

- `val_wer` is the main quality metric for the RNNT decode path.
- `val_wer_ctc` is the quality metric for the CTC branch.
- Lower is better for both.

What to notice:

- The most important trend is whether `val_wer` keeps going down across validations.
- If `train_loss` drops but `val_wer` stalls or rises, the run is overfitting or learning the wrong thing.
- If `val_wer_ctc` improves while `val_wer` does not, the auxiliary branch is learning more than the main RNNT decode path.
- If both stay very high, check data quality, tokenization, language IDs, and whether the fine-tuning run is too short.

Healthy shape:

- A noisy but generally downward trend.

Concerning shape:

- Flat, rising, or a sharp improvement followed by steady regression.

### Training Batch WER

Tags:

- `training_batch_wer`
- `training_batch_wer_ctc`

Purpose:

- These are quick batch-level training signals, not final model-quality metrics.
- They are useful for spotting whether the model is learning at all.

What to notice:

- Expect these to be noisy because they come from individual training batches.
- The average direction should still move downward over time.
- Large oscillations are normal with `batch_size=1`, but persistent saturation near `1.0` is a bad sign.
- If training batch WER looks great but validation WER stays poor, the model is memorizing the train batches.

Healthy shape:

- Noisy but gradually lower peaks and lower typical values.

Concerning shape:

- Stuck near `1.0`, or collapsing to `0` while validation stays bad.

### Loss Comparison

Tags:

- `train_loss`
- `train_rnnt_loss`
- `train_ctc_loss`

Purpose:

- `train_loss` is the optimization target being minimized.
- `train_rnnt_loss` is the RNNT component.
- `train_ctc_loss` is the auxiliary CTC component.

What to notice:

- All three should usually trend downward overall.
- The absolute values matter less than the direction and stability.
- If one branch falls and the other does not, that branch may be dominating or lagging.
- Sudden spikes often point to unstable batches, LR issues, or bad samples.
- If loss keeps falling but `val_wer` does not improve, the model is optimizing training objective without real transcription gains.

Healthy shape:

- Fast early drop, then slower improvement.

Concerning shape:

- Repeated spikes, long flat plateaus, or divergence upward.

### Learning Rate

Tag:

- `learning_rate`

Purpose:

- Shows the optimizer schedule that drives update size.

What to notice:

- Verify it matches the schedule you intended.
- If it is too small for most of the run, learning may be unnecessarily slow.
- If it jumps unexpectedly, check scheduler configuration and warmup settings.
- Compare LR transitions with loss spikes or validation regressions.

Healthy shape:

- Smooth curve matching your scheduler design.

Concerning shape:

- Near-zero almost everywhere, or sudden discontinuities you did not plan.

### Progress

Tags:

- `global_step`
- `epoch`

Purpose:

- Confirms whether training advanced as expected.
- Helps interpret where validation events happened relative to the run.

What to notice:

- `global_step` should increase steadily.
- `epoch` should increment when a full pass finishes.
- If `epoch` stays at `0`, the run ended before finishing the first epoch.
- If validation logged only once, you likely validated only at epoch end or the run stopped early.

Healthy shape:

- Monotonic progress with validations appearing at expected intervals.

Concerning shape:

- Short runs, missing validation points, or progress stopping unexpectedly.

### Train Timing

Tags:

- `train_step_timing in s`
- `train_backward_timing in s`

Purpose:

- These track training speed and help identify bottlenecks.

What to notice:

- A stable range usually means the input pipeline and GPU usage are behaving consistently.
- Gradual improvement at the start is normal due to warmup and caching.
- Large spikes can indicate data-loading stalls, host-memory pressure, or GPU contention.
- If backward time grows much more than step time used to, look for instability or memory pressure.

Healthy shape:

- Mostly stable with minor jitter.

Concerning shape:

- Frequent large spikes or a drifting upward trend.

### Validation Timing

Tag:

- `validation_step_timing in s`

Purpose:

- Measures validation-step latency.

What to notice:

- This is mainly operational, not model quality.
- Use it to spot data-loader stalls or slow decode behavior during validation.
- If validation timing worsens sharply after a config change, inspect decoder settings and host load.

Healthy shape:

- Stable or slightly improving.

Concerning shape:

- Wide latency swings or sustained slowdown.

## Best Practice For Better TensorBoard Visibility

For future runs, prefer step-based validation instead of only epoch-end validation:

- set `trainer.val_check_interval` to a step count such as `400`
- keep `trainer.log_every_n_steps` reasonably small such as `25`
- keep `exp_manager.checkpoint_callback_params.monitor=val_wer`
- keep early stopping on `val_wer`

This makes the important graph, `val_wer`, much easier to read and much more useful during training.

## Notes For Your Current Run

For the run at `2026-04-08_13-23-58`:

- the event file contains `train_loss`, `train_rnnt_loss`, `train_ctc_loss`, `val_wer`, `val_wer_ctc`, `learning_rate`, timing metrics, and batch WER metrics
- it logged only one validation point
- `epoch` stayed at `0`
- the checkpoint filename includes `unfinished`

That means the layout can be improved immediately, but for truly readable validation curves you will also want step-based validation on the next run.
