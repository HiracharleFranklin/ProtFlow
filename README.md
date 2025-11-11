# ProtFlow

## Training

### 1. Environment creation

```
conda env create -f environment.yaml
```

### 2. Compute statistics

```
python -m utils.get_statistics
python -m utils.stat
```

### 3. Decoder training

```
python train_decoder.py
```

### 4. Diffusion model training

```
torchrun --nproc_per_node=1 --master_port=31345  train_flow_matching.py
```

## Generation

```
torchrun --nproc_per_node=1 --master_port=31345  generation.py
```
