"""
Dataset registration for continual learning in VITA.
Adapted from HVPL hvpl/data/datasets/builtin.py
"""
import os
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata

from .coco_increment import register_coco_instances_incremental
from .ytvis_increment import (
    register_ytvis_instances_incremental,
    _get_ytvis_2019_instances_meta,
    _get_ytvis_2021_instances_meta
)
from .ovis_increment import _get_ovis_instances_meta

# Predefined splits for YTVIS 2019
_PREDEFINED_SPLITS_YTVIS_2019 = {
    "ytvis_2019_train": ("ytvis_2019/train/JPEGImages",
                         "ytvis_2019/train.json"),
    "ytvis_2019_val": ("ytvis_2019/valid/JPEGImages",
                       "ytvis_2019/valid.json"),
    "ytvis_2019_test": ("ytvis_2019/test/JPEGImages",
                        "ytvis_2019/test.json"),
}

# Predefined splits for YTVIS 2021
_PREDEFINED_SPLITS_YTVIS_2021 = {
    "ytvis_2021_train": ("ytvis_2021/train/JPEGImages",
                         "ytvis_2021/train.json"),
    "ytvis_2021_val": ("ytvis_2021/valid/JPEGImages",
                       "ytvis_2021/valid.json"),
    "ytvis_2021_test": ("ytvis_2021/test/JPEGImages",
                        "ytvis_2021/test.json"),
}

# Predefined splits for OVIS
_PREDEFINED_SPLITS_OVIS = {
    "ovis_train": ("ovis/train",
                   "ovis/annotations_train.json"),
    "ovis_val": ("ovis/valid",
                 "ovis/annotations_valid.json"),
    "ovis_test": ("ovis/test",
                  "ovis/annotations_test.json"),
}

# COCO to YTVIS/OVIS mappings
_PREDEFINED_SPLITS_COCO_VIDEO = {
    "coco2ytvis2019_train": ("coco/train2017", "coco/annotations/coco2ytvis2019_train.json"),
    "coco2ytvis2019_val": ("coco/val2017", "coco/annotations/coco2ytvis2019_val.json"),
    "coco2ytvis2021_train": ("coco/train2017", "coco/annotations/coco2ytvis2021_train.json"),
    "coco2ytvis2021_val": ("coco/val2017", "coco/annotations/coco2ytvis2021_val.json"),
    "coco2ovis_train": ("coco/train2017", "coco/annotations/coco2ovis_train.json"),
    "coco2ovis_val": ("coco/val2017", "coco/annotations/coco2ovis_val.json"),
}


def register_all_ytvis_2019_incremental(root, cfg, train=True):
    """Register YTVIS 2019 datasets with incremental learning support."""
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2019.items():
        register_ytvis_instances_incremental(
            key,
            _get_ytvis_2019_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            cfg,
            train,
        )


def register_all_ytvis_2021_incremental(root, cfg, train=True):
    """Register YTVIS 2021 datasets with incremental learning support."""
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2021.items():
        register_ytvis_instances_incremental(
            key,
            _get_ytvis_2021_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            cfg,
            train,
        )


def register_all_ovis_incremental(root, cfg, train=True):
    """Register OVIS datasets with incremental learning support."""
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_OVIS.items():
        register_ytvis_instances_incremental(
            key,
            _get_ovis_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            cfg,
            train,
        )


def register_all_coco_video_incremental(root, cfg):
    """Register COCO video datasets with incremental learning support."""
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_COCO_VIDEO.items():
        register_coco_instances_incremental(
            key,
            _get_builtin_metadata("coco"),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            cfg,
        )
