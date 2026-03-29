# Continual Learning for VITA

This directory contains the continual learning implementation for VITA, adapted from HVPL.

## Overview

The continual learning module enables VITA to learn new object classes incrementally without forgetting previously learned classes. This is essential for real-world applications where new categories emerge over time.

## Key Components

### 1. Configuration (`vita/continual_config.py`)
- `CONT.BASE_CLS`: Number of base classes (e.g., 20)
- `CONT.INC_CLS`: Number of incremental classes per task (e.g., 2)
- `CONT.TASK`: Current task ID (0 = base task)
- `CONT.MODE`: Data mode ("overlap", "disjoint", "sequential")

### 2. Data Processing
- `vita/data/ytvis_increment.py`: YTVIS dataset with class filtering
- `vita/data/coco_increment.py`: COCO dataset with class filtering
- `vita/data/ovis_increment.py`: OVIS dataset support
- `vita/data/builtin_continual.py`: Dataset registration

### 3. Training
- `train_net_vita_continual.py`: Main training script

### 4. Evaluation
- `vita/evaluation/continual_eval.py`: Forgetting metrics computation

## Usage

### Task 0 (Base Task - 20 classes)
```bash
python train_net_vita_continual.py \
  --config-file configs/ytvis_2019_continual.yaml \
  --num-gpus 8 \
  CONT.TASK 0 \
  CONT.BASE_CLS 20 \
  CONT.INC_CLS 2
```

### Task 1 (Incremental Task - 22 classes total)
```bash
python train_net_vita_continual.py \
  --config-file configs/ytvis_2019_continual.yaml \
  --num-gpus 8 \
  CONT.TASK 1 \
  CONT.BASE_CLS 20 \
  CONT.INC_CLS 2
```

### Task 2 (Incremental Task - 24 classes total)
```bash
python train_net_vita_continual.py \
  --config-file configs/ytvis_2019_continual.yaml \
  --num-gpus 8 \
  CONT.TASK 2 \
  CONT.BASE_CLS 20 \
  CONT.INC_CLS 2
```

## Continual Learning Modes

### 1. Overlap Mode (Default)
- Images containing new classes are used for training
- Most flexible, allows mixed class images

### 2. Disjoint Mode
- Images only contain current task classes + old classes
- No future class annotations in training data

### 3. Sequential Mode
- Images can contain all learned classes
- Suitable for sequential learning scenarios

## Data Filtering

The incremental learning system automatically filters annotations based on the current task:

**Task 0 (Base)**: Classes 0-19
**Task 1**: Classes 20-21 (new)
**Task 2**: Classes 22-23 (new)
...

## COCO to YTVIS/OVIS Mapping

The system includes category mappings for using COCO as auxiliary data:

- `COCO_TO_YTVIS_2019`: 21 COCO categories → YTVIS 2019
- `COCO_TO_YTVIS_2021`: 23 COCO categories → YTVIS 2021
- `COCO_TO_OVIS`: COCO categories → OVIS

## Evaluation Metrics

### Standard Metrics
- **AP**: Average Precision
- **AP50**: AP at IoU=0.50
- **AP75**: AP at IoU=0.75
- **AR1**: Average Recall with 1 detection per image
- **AR10**: Average Recall with 10 detections per image

### Continual Learning Metrics
- **FAP**: Forgetting of Average Precision
- **FAR**: Forgetting of Average Recall

Forgetting is computed as:
```
Forgetting = (Best_Performance_When_Learned - Current_Performance) / Best_Performance_When_Learned
```

## File Structure

```
VITA-main/
├── train_net_vita_continual.py          # Main training script
├── vita/
│   ├── continual_config.py              # Continual learning config
│   ├── data/
│   │   ├── ytvis_increment.py           # YTVIS incremental loader
│   │   ├── coco_increment.py            # COCO incremental loader
│   │   ├── ovis_increment.py            # OVIS support
│   │   └── builtin_continual.py         # Dataset registration
│   └── evaluation/
│       └── continual_eval.py            # Evaluation metrics
└── configs/
    └── ytvis_2019_continual.yaml        # Example config
```

## Integration with MoE

This continual learning framework is designed to work with Mixture of Experts (MoE):

1. **Task-specific Experts**: Each task can activate different expert combinations
2. **Expert Routing**: Route based on task ID or learned features
3. **Capacity Management**: Balance expert usage across tasks
4. **Forgetting Prevention**: Experts preserve old task knowledge

## Next Steps

1. Create config files for your specific scenarios
2. Prepare dataset splits (train/val/test)
3. Run base task training (Task 0)
4. Run incremental tasks sequentially
5. Evaluate forgetting metrics

## References

- VITA: https://github.com/sukjunhwang/VITA
- HVPL: Hierarchical Visual Prompt Learning (ICCV 2025)
