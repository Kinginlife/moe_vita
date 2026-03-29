"""
MoE Continual Learning Training Script for VITA.

Usage:
    # Task 0 (Base task)
    python train_net_vita_moe.py --config-file configs/ytvis_2019_moe.yaml \
        --num-gpus 8 CONT.TASK 0 CONT.BASE_CLS 20 CONT.INC_CLS 2 MOE.NUM_EXPERTS 1

    # Task 1 (First incremental task)
    python train_net_vita_moe.py --config-file configs/ytvis_2019_moe.yaml \
        --num-gpus 8 CONT.TASK 1 CONT.BASE_CLS 20 CONT.INC_CLS 2 MOE.NUM_EXPERTS 2
"""
import os
import logging

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch

from train_net_vita_continual import ContinualTrainer, setup
from vita.continual_config import add_continual_config


logger = logging.getLogger("detectron2")


class MoETrainer(ContinualTrainer):
    """Trainer for MoE continual learning."""

    @classmethod
    def build_model(cls, cfg):
        model = super().build_model(cfg)

        # Handle MoE expert management for incremental tasks
        if cfg.MOE.ENABLED and cfg.CONT.TASK > 0:
            # Get the MoE layer (last FFN layer in decoder)
            decoder = model.sem_seg_head.predictor
            moe_layer = decoder.transformer_ffn_layers[-1].moe

            # Freeze old experts
            if cfg.MOE.FREEZE_OLD_EXPERTS:
                old_expert_ids = list(range(cfg.CONT.TASK))
                moe_layer.freeze_experts(old_expert_ids)
                logger.info(f"Frozen experts: {old_expert_ids}")

        return model



def main(args):
    cfg = setup(args)

    # Update MoE config based on task
    if cfg.MOE.ENABLED:
        cfg.defrost()
        cfg.MOE.NUM_EXPERTS = cfg.CONT.TASK + 1
        cfg.freeze()
        logger.info(f"Task {cfg.CONT.TASK}: Using {cfg.MOE.NUM_EXPERTS} experts")

    if args.eval_only:
        model = MoETrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = MoETrainer.test(cfg, model)
        return res

    trainer = MoETrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
