"""
Compute OGP projection matrix from Task 0 features.
Run this after Task 0 training completes, before starting Task 1.

Usage:
    python tools/compute_ogp_projection.py \
        --config-file configs/your_config.yaml \
        --task 0 \
        --output-dir output/task0 \
        --num-samples 1000
"""
import argparse
import os
import torch
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.projects.deeplab import add_deeplab_config

from mask2former import add_maskformer2_config
from vita import add_vita_config, YTVISDatasetMapper
from vita.continual_config import add_continual_config

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from train_net_vita import Trainer


def collect_router_features(model, data_loader, num_samples=1000):
    """Collect router input features from training data."""
    model.eval()
    features_list = []

    # Hook to capture router input
    router_inputs = []
    def hook_fn(module, input, output):
        router_inputs.append(input[0].detach().cpu())

    # Register hook on router
    decoder = model.sem_seg_head.predictor
    moe_layers = [layer for layer in decoder.transformer_ffn_layers if hasattr(layer, 'moe')]
    if len(moe_layers) == 0:
        raise ValueError("No MoE layers found!")

    hook = moe_layers[0].moe.router.register_forward_hook(hook_fn)

    with torch.no_grad():
        for idx, inputs in enumerate(tqdm(data_loader, desc="Collecting features")):
            if idx >= num_samples:
                break

            images = []
            for video in inputs:
                for frame in video["image"]:
                    images.append(frame.to(model.device))

            from detectron2.structures import ImageList
            images = [(x - model.pixel_mean) / model.pixel_std for x in images]
            images = ImageList.from_tensors(images, model.size_divisibility)

            backbone_features = model.backbone(images.tensor)
            _ = model.sem_seg_head(backbone_features)

    hook.remove()

    # Concatenate all captured features: [N, D]
    all_features = torch.cat([f.flatten(0, 1) for f in router_inputs], dim=0)
    return all_features


def main():
    parser = argparse.ArgumentParser(description="Compute OGP projection matrix")
    parser.add_argument("--config-file", required=True, help="path to config file")
    parser.add_argument("--task", type=int, default=0, help="task id")
    parser.add_argument("--output-dir", required=True, help="output directory")
    parser.add_argument("--num-samples", type=int, default=1000, help="number of samples")
    parser.add_argument("--energy-threshold", type=float, default=0.95, help="energy threshold")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # Setup config
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_vita_config(cfg)
    add_continual_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # Build model
    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()

    # Build TRAIN data loader (not test!)
    from vita import YTVISDatasetMapper, CocoClipDatasetMapper
    dataset_name = cfg.DATASETS.TRAIN[0]
    if dataset_name.startswith('coco'):
        mapper = CocoClipDatasetMapper(cfg, is_train=False)
    else:
        mapper = YTVISDatasetMapper(cfg, is_train=False)

    from detectron2.data import build_detection_test_loader
    data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    print(f"Collecting features from {args.num_samples} samples...")
    features = collect_router_features(model, data_loader, args.num_samples)
    print(f"Collected features shape: {features.shape}")

    # Compute projection matrix
    print("Computing projection matrix...")
    decoder = model.sem_seg_head.predictor
    moe_layers = [layer for layer in decoder.transformer_ffn_layers if hasattr(layer, 'moe')]

    if len(moe_layers) == 0:
        print("No MoE layers found!")
        return

    router = moe_layers[0].moe.router

    # === 修改后的代码：正确寻找上一个 Task 的文件夹 ===
    if args.task > 0:
        output_base = os.path.dirname(args.output_dir) # 退回到 output/ytvis_2019_moe_hvpl
        prev_task = args.task - 1
        prev_proj_path = os.path.join(output_base, f"step{prev_task}", f"projection_matrix_task{prev_task}.pt")
    else:
        prev_proj_path = None

    if prev_proj_path and os.path.exists(prev_proj_path):
        print(f"Loading previous projection matrix from {prev_proj_path}")
        prev_proj = torch.load(prev_proj_path, map_location=model.device)
        # Apply previous projection to current features to get orthogonal component
        features = torch.matmul(features.to(model.device), prev_proj)
        print("Applied previous projection to current features")
        
    router.compute_projection_matrix(features.to(model.device), args.energy_threshold)

    # Combine with previous projection if exists
    if prev_proj_path and os.path.exists(prev_proj_path):
        # P_new = P_prev @ P_current (chain projections)
        # 确保两者在同一个 device 上进行矩阵乘法
        router.projection_matrix = torch.matmul(prev_proj.to(model.device), router.projection_matrix.to(model.device))
        print("Combined with previous projection matrix")

    # Save projection matrix
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"projection_matrix_task{args.task}.pt")
    torch.save(router.projection_matrix.cpu(), output_path)
    print(f"Saved projection matrix to {output_path}")
    print(f"Matrix shape: {router.projection_matrix.shape}")


if __name__ == "__main__":
    main()
