# VITA 持续学习训练脚本说明

## 📁 脚本目录结构

```
VITA-main/scripts/
├── ytvis_2019_20_2.sh    # YouTube-VIS 2019: 20+2 场景
├── ytvis_2019_20_5.sh    # YouTube-VIS 2019: 20+5 场景
├── ytvis_2021_20_4.sh    # YouTube-VIS 2021: 20+4 场景
└── ovis_15_5.sh          # OVIS: 15+5 场景
```

## 🚀 使用方法

### 1. YouTube-VIS 2019 (20+2)
```bash
cd /Users/lsh21/Downloads/vitamoe/VITA-main
bash scripts/ytvis_2019_20_2.sh
```

**场景**: 20个基础类 + 每次增加2个类
**任务**: 11个任务 (Task 0-10)
**总类别**: 40类
**训练迭代**:
- Task 0: 80,000 iterations
- Task 1-10: 10,000 iterations each

### 2. YouTube-VIS 2019 (20+5)
```bash
bash scripts/ytvis_2019_20_5.sh
```

**场景**: 20个基础类 + 每次增加5个类
**任务**: 5个任务 (Task 0-4)
**训练迭代**:
- Task 0: 80,000 iterations
- Task 1-4: 30,000 iterations each

### 3. YouTube-VIS 2021 (20+4)
```bash
bash scripts/ytvis_2021_20_4.sh
```

**场景**: 20个基础类 + 每次增加4个类
**任务**: 6个任务 (Task 0-5)
**训练迭代**:
- Task 0: 80,000 iterations
- Task 1-5: 15,000 iterations each

### 4. OVIS (15+5)
```bash
bash scripts/ovis_15_5.sh
```

**场景**: 15个基础类 + 每次增加5个类
**任务**: 3个任务 (Task 0-2)
**训练迭代**:
- Task 0: 80,000 iterations
- Task 1-2: 30,000 iterations each

## ⚙️ 关键参数说明

### 评估和保存频率
- **EVAL_PERIOD**: 评估周期（iterations）
  - Base task: 5000
  - Incremental tasks: 2000-5000
- **CHECKPOINT_PERIOD**: Checkpoint保存周期
  - Base task: 5000
  - Incremental tasks: 2000-5000

### GPU配置
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NGPUS=8
```

### 数据集路径
```bash
export DETECTRON2_DATASETS=datasets
```

## 📊 输出文件结构

```
output/
└── ytvis_2019_continual/
    ├── ytvis_2019_train_20-2-ov/
    │   └── ytvis_2019_20_2/
    │       ├── task0/
    │       │   ├── model_final.pth
    │       │   ├── metrics_task0.txt
    │       │   └── inference/
    │       │       └── results.json
    │       ├── task1/
    │       │   ├── model_final.pth
    │       │   ├── metrics_task1.txt
    │       │   └── inference/
    │       ├── results_0.json  # 自动保存
    │       ├── results_1.json
    │       └── ...
```

## ✅ 自动保存的文件

1. **results_{task_id}.json**: 每个任务的预测结果（COCO格式）
2. **metrics_task{task_id}.txt**: 每个任务的评估指标
3. **model_final.pth**: 每个任务的最终模型权重
4. **model_{iter}.pth**: 定期保存的checkpoint

## 🔄 任务间权重继承

脚本会自动加载上一任务的权重：
```python
# Task 1 自动加载 Task 0 的权重
# Task 2 自动加载 Task 1 的权重
# ...
```

## 📝 修改脚本

### 修改GPU数量
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 使用4个GPU
NGPUS=4
```

### 修改训练迭代数
```bash
ITER_BASE=100000      # 基础任务迭代数
ITER_INC=20000        # 增量任务迭代数
```

### 修改评估频率
```bash
TEST.EVAL_PERIOD 2000           # 每2000次迭代评估一次
SOLVER.CHECKPOINT_PERIOD 2000   # 每2000次迭代保存一次
```

## 🎯 与HVPL的对齐

✅ **训练流程**: 完全对齐
✅ **评估周期**: 完全对齐
✅ **Checkpoint保存**: 完全对齐
✅ **结果文件**: 自动保存为results_{task_id}.json
✅ **指标保存**: 自动保存为metrics_task{task_id}.txt
