# 测试指南 - VITA 增量学习数据处理

## 快速测试

### 方法1: 使用测试脚本
```bash
cd /Users/lsh21/Downloads/vitamoe/VITA-main
bash run_test.sh
```

### 方法2: 直接运行Python测试
```bash
cd /Users/lsh21/Downloads/vitamoe/VITA-main
python test_continual_data.py
```

## 测试内容

### Test 1: COCO增量加载测试
验证COCO数据集的类别过滤逻辑：
- **Task 0**: 只加载类别 0-19 (基础类别)
- **Task 1**: 只加载类别 20-21 (新增类别)
- **Task 2**: 只加载类别 22-23 (新增类别)

### Test 2: 类别映射测试
验证COCO到YTVIS/OVIS的类别映射：
- COCO_TO_YTVIS_2019: 21个映射关系
- COCO_TO_YTVIS_2021: 23个映射关系

### Test 3: 元数据测试
验证YTVIS 2019/2021的元数据：
- 类别数量: 40个
- 类别名称列表
- 颜色映射

## 预期输出

```
==========================================================
VITA Continual Learning Data Processing Tests
==========================================================

==========================================================
TEST 1: COCO Incremental Loading
==========================================================
✓ Created mock COCO JSON: /tmp/.../mock_coco.json
  - 10 images
  - 25 annotations
  - 40 categories

--- Task 0: Base Classes (0-19) ---
✓ Loaded X images for Task 0
  Categories found: [0, 1, 2, ..., 19]
  Expected: 0-19 (base classes)

--- Task 1: Incremental Classes (20-21) ---
✓ Loaded Y images for Task 1
  Categories found: [20, 21]
  Expected: 20-21 (new classes)

✅ COCO incremental loading test PASSED

==========================================================
TEST 2: Category Mappings
==========================================================
Total mappings: 21
Sample mappings:
  COCO 1 → YTVIS 1
  COCO 2 → YTVIS 21
  ...

✅ Category mapping test PASSED

==========================================================
TEST 3: Metadata
==========================================================
Number of classes: 40
Sample classes: ['person', 'giant_panda', 'lizard', ...]

✅ Metadata test PASSED

==========================================================
✅ ALL TESTS PASSED
==========================================================
```

## 测试验证点

✅ **类别过滤正确性**
- Task 0 只包含基础类别
- Task 1 只包含新增类别
- 没有未来类别泄露

✅ **类别映射完整性**
- COCO到YTVIS的映射关系正确
- 映射数量符合预期

✅ **元数据一致性**
- 类别数量正确 (YTVIS: 40, OVIS: 25)
- 类别名称和ID对应正确

## 故障排查

### 如果测试失败

1. **ImportError**: 确保在VITA-main目录下运行
2. **ModuleNotFoundError**: 检查detectron2是否安装
3. **类别过滤错误**: 检查cfg.CONT配置是否正确

### 调试模式
```python
# 在test_continual_data.py中添加调试输出
print(f"Debug: cfg.CONT.TASK = {cfg.CONT.TASK}")
print(f"Debug: cfg.CONT.BASE_CLS = {cfg.CONT.BASE_CLS}")
print(f"Debug: cfg.CONT.INC_CLS = {cfg.CONT.INC_CLS}")
```

## 下一步

测试通过后，可以：
1. 准备真实数据集
2. 运行完整训练流程
3. 集成MoE架构
