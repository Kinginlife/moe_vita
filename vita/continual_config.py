# -*- coding: utf-8 -*-
"""
Continual Learning Configuration for VITA
Adapted from HVPL continual/config.py
"""
from detectron2.config import CfgNode as CN


def add_continual_config(cfg):
    """
    Add config for continual learning in VITA.
    """
    cfg.CONT = CN()

    # Basic continual learning settings
    cfg.CONT.BASE_CLS = 20              # Number of base classes
    cfg.CONT.INC_CLS = 2                # Number of incremental classes per task
    cfg.CONT.TASK = 0                   # Current task ID (0 = base task)
    cfg.CONT.ORDER = list(range(1, 41)) # Class learning order (1-40 for YTVIS)
    cfg.CONT.ORDER_NAME = None          # Optional name for class order
    cfg.CONT.WEIGHTS = None             # Path to pretrained weights

    # Continual learning mode
    # "overlap": images with new classes are used
    # "disjoint": images only with current+old classes (no future classes)
    # "sequential": images with all learned classes
    cfg.CONT.MODE = "overlap"

    # Model configuration
    cfg.CONT.FREEZE_BACKBONE = False    # Whether to freeze backbone
    cfg.CONT.FREEZE_DETECTOR = False    # Whether to freeze detector

    # Knowledge distillation settings
    cfg.CONT.DIST = CN()
    cfg.CONT.DIST.KD_WEIGHT = 0.0       # Knowledge distillation weight
    cfg.CONT.DIST.POD_WEIGHT = 0.0      # Pooled output distillation weight

    # Dataset settings
    cfg.DATASETS.DATASET_RATIO = []     # Ratio for mixing multiple datasets

    # MoE settings
    cfg.MOE = CN()
    cfg.MOE.ENABLED = False             # Enable MoE
    cfg.MOE.NUM_EXPERTS = 1             # Number of experts (grows with tasks)
    cfg.MOE.ROUTER_DIM = 512            # Router hidden dimension
    cfg.MOE.TOP_K = 1                   # Number of experts to activate
    cfg.MOE.ROUTING_LOSS_WEIGHT = 0.1   # Weight for routing loss
    cfg.MOE.FREEZE_OLD_EXPERTS = True   # Freeze old experts in incremental tasks
    cfg.MOE.INIT_FROM_PREVIOUS = True   # Initialize new expert from previous
