# ProtFlow

## Environment creation

```
conda env create -f environment.yaml
```

Ensure all paths and hyperparameters in this project are defined according to your specific tasks. Download facebook/esm2_t12_35M_UR50D, Rostlab/prot_t5_xl_uniref50 and Rostlab/prot_t5_xl_half_uniref50-enc from hunggingface and add them at the root folder.

## Training

### 1. Compute statistics

```
python -m utils.get_statistics
python -m utils.stat
```

### 2. Decoder training

```
python train_decoder.py
```

### 3. Compression-decompression module training

```
python train_compressor.py
```

### 4. FM holder training

```
torchrun --nproc_per_node=1 --master_port=31345  train_flow_matching.py
```

## Generation

```
torchrun --nproc_per_node=1 --master_port=31345  generation.py
```
