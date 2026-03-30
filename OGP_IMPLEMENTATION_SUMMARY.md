# OGP 实现总结

## 📋 修改文件清单

### 核心实现
1. **vita/modeling/moe/router.py** - OGP 核心逻辑
2. **vita/modeling/moe/moe_layer.py** - 新专家权重初始化投影
3. **vita/modeling/transformer_decoder/vita_moe_decoder.py** - 分类头旧类冻结
4. **vita/vita_model.py** - 彻底冻结底层特征
5. **vita/modeling/vita_criterion.py** - 伪标签防背景偏移

### 训练流程
6. **train_net_vita.py** - 训练初始化和投影矩阵加载
7. **tools/compute_ogp_projection.py** - 特征收集和投影矩阵计算（新文件）
8. **scripts/ytvis_2019_20_2_moe.sh** - 自动化训练脚本

---

## 🎯 四重防遗忘机制

### 1. 特征冻结（Feature Freezing）
**位置**: `vita/vita_model.py` 第 119-147 行

**冻结内容**:
- Backbone（完全冻结）
- Pixel Decoder（完全冻结）
- Query Embeddings（query_embed, query_feat）
- 所有 Attention 层（Self-Attention, Cross-Attention）
- 前 N-1 层 FFN

**允许训练**:
- 最后一层 MoE（新专家 + Router）
- 分类头 class_embed

**关键代码**:
```python
def set_task_id(self, task_id):
    if task_id > 0:
        # 冻结所有层，只保留最后 MoE 和 class_embed
        for name, p in model.named_parameters():
            if not (is_last_moe or 'class_embed' in name):
                p.requires_grad = False
```

---

### 2. 正交梯度投影（OGP）
**位置**: `vita/modeling/moe/router.py` 第 29-75 行

**机制**:
