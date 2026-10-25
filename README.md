# Action tokenization (DROID)

FlowVQ and a k-means initialized VQ-VAE tokenizer for chunks of 50 robot actions
(7-D: 6-DoF end-effector + gripper) from the [DROID](https://droid-dataset.github.io/) dataset.

## Requirements

```bash
pip install -r requirements.txt
```

## Data

Download DROID in RLDS format so that `tensorflow_datasets` finds it as `data/droid`
(i.e. `tfds.load("droid", data_dir="data")`). On first use, `dataset.py` keeps the successful episodes,
splits them randomly 80/20 into train/test and caches the actions in `data/droid/parsed_data.pt`.
Later runs only read this cache. The split is random, so use the same `parsed_data.pt` to compare runs.

## Training

```bash
# FlowVQ tokenizer
python main.py
# K-means initialized VQ-VAE baseline
python main.py --quantizer kmeans
```

The tokenizer is trained as an auto-encoder for 1000 steps, then its codebook (K=2048, d=8) is
initialized from the encoder latents (flow matching for FlowVQ, k-means for the baseline) and training
continues with vector quantization. Checkpoints are written to `output/<run_name>/`:

- `model_ae_biattn_<step>.pt`: the auto-encoder right before codebook initialization. Pass it with
  `--ckpt` to restart from that point with a fresh codebook.
- `model_ae_biattn_init.pt`: FlowVQ model right after codebook initialization.
- `model_e<epoch>.pt`: model at the end of every epoch.

Use `--no_wandb` to disable Weights & Biases logging, and `--gpu` to choose the device.
All hyper-parameters are defined at the top of `main.py`.

## Evaluation

Reconstruction MSE and codebook usage on the test split:

```bash
python eval.py output/<run_name>/model_e<epoch>.pt
```

