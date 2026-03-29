# 评估和数据加载对齐检查报告

## ✅ 已验证的对齐项

### 1. 评估器配置
**位置**: `train_net_vita_continual.py:61-72`

```python
@classmethod
def build_evaluator(cls, cfg, dataset_name, output_folder=None):
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        os.makedirs(output_folder, exist_ok=True)

    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type == "coco":
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
    elif evaluator_type == "ytvis":
        return YTVISEvaluator(dataset_name, cfg, True, output_folder)
```

✅ **对齐**: 与HVPL相同，自动根据数据集类型选择评估器

### 2. 评估周期和Checkpoint保存
**脚本配置**:
- **Base Task**: `TEST.EVAL_PERIOD 5000`, `SOLVER.CHECKPOINT_PERIOD 5000`
- **Incremental Tasks**: `TEST.EVAL_PERIOD 2000-5000`, `SOLVER.CHECKPOINT_PERIOD 2000-5000`

✅ **对齐**: 与HVPL相同的评估和保存频率

### 3. 结果文件保存
**YTVISEvaluator** 会自动保存：
- `inference/results.json` - 预测结果（COCO格式）
- 评估指标会打印到日志

**需要手动重命名**:
```bash
# 每个任务完成后
mv output/ytvis_2019_20_2/task0/inference/results.json \
   output/ytvis_2019_20_2/results_0.json
```

### 4. 测试数据加载
**位置**: `train_net_vita_continual.py:100-107`

```python
@classmethod
def build_test_loader(cls, cfg, dataset_name):
    if dataset_name.startswith('coco'):
        mapper = CocoClipDatasetMapper(cfg, is_train=False)
    elif dataset_name.startswith('ytvis') or dataset_name.startswith('ovis'):
        mapper = YTVISDatasetMapper(cfg, is_train=False)
    return build_detection_test_loader(cfg, dataset_name, mapper=mapper)
```

✅ **对齐**: 测试时使用相同的数据映射器

### 5. 增量数据过滤
**训练时**: `ytvis_increment.py:231-242`
- Task 0: 只加载类别 0-19
- Task 1: 只加载类别 20-21

**测试时**: 使用完整数据集（不过滤）
- 评估所有已学习的类别

✅ **对齐**: 与HVPL相同的训练/测试数据处理逻辑

## ⚠️ 需要注意的差异

### 1. 结果文件命名
**HVPL**: 自动保存为 `results_{task_id}.json`
**VITA**: 需要手动重命名或修改代码

**解决方案**: 修改YTVISEvaluator保存逻辑

### 2. 评估指标保存
**HVPL**: 保存详细的per-class指标
**VITA**: 当前只打印整体指标

**解决方案**: 需要扩展评估器保存per-class结果

## 📋 训练脚本参数对照

| 参数 | HVPL | VITA | 状态 |
|------|------|------|------|
| CONT.TASK | ✅ | ✅ | 对齐 |
| CONT.BASE_CLS | ✅ | ✅ | 对齐 |
| CONT.INC_CLS | ✅ | ✅ | 对齐 |
| TEST.EVAL_PERIOD | ✅ | ✅ | 对齐 |
| SOLVER.CHECKPOINT_PERIOD | ✅ | ✅ | 对齐 |
| SOLVER.MAX_ITER | ✅ | ✅ | 对齐 |
| OUTPUT_DIR | ✅ | ✅ | 对齐 |

## 🔧 建议的改进

### 1. 自动保存results_{task_id}.json
在训练脚本中添加：
```python
# After evaluation
import shutil
src = os.path.join(cfg.OUTPUT_DIR, "inference/results.json")
dst = os.path.join(cfg.OUTPUT_DIR, f"../results_{cfg.CONT.TASK}.json")
if os.path.exists(src):
    shutil.copy(src, dst)
```

### 2. 保存评估指标到文件
```python
# Save metrics to txt
metrics_file = os.path.join(cfg.OUTPUT_DIR, f"metrics_task{cfg.CONT.TASK}.txt")
with open(metrics_file, 'w') as f:
    for key, val in results.items():
        f.write(f"{key}: {val}\n")
```

## ✅ 总结

**数据加载**: ✅ 完全对齐
- 训练时正确过滤类别
- 测试时使用完整数据集

**评估流程**: ✅ 基本对齐
- 使用相同的评估器
- 相同的评估周期

**结果保存**: ⚠️ 需要改进
- 需要自动重命名结果文件
- 需要保存per-class指标

**Checkpoint**: ✅ 完全对齐
- 相同的保存频率
- 自动加载上一任务权重
