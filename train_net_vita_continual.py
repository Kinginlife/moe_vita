"""
Continual Learning Training Script for VITA.
Adapted from HVPL train_net_hvpl.py

Usage:
    # Task 0 (Base task)
    python train_net_vita_continual.py --config-file configs/ytvis_2019_continual.yaml \
        --num-gpus 8 CONT.TASK 0 CONT.BASE_CLS 20 CONT.INC_CLS 2

    # Task 1 (First incremental task)
    python train_net_vita_continual.py --config-file configs/ytvis_2019_continual.yaml \
        --num-gpus 8 CONT.TASK 1 CONT.BASE_CLS 20 CONT.INC_CLS 2
"""
import os
import logging

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    COCOEvaluator,
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from vita import (
    YTVISDatasetMapper,
    CocoClipDatasetMapper,
    YTVISEvaluator,
    build_combined_loader,
    build_detection_train_loader,
    build_detection_test_loader,
    add_vita_config,
)
from vita.continual_config import add_continual_config
from vita.data.builtin_continual import (
    register_all_ytvis_2019_incremental,
    register_all_ytvis_2021_incremental,
    register_all_ovis_incremental,
    register_all_coco_video_incremental,
)

logger = logging.getLogger("detectron2")


class ContinualTrainer(DefaultTrainer):
    """Trainer for continual learning."""

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
        else:
            raise NotImplementedError(f"No evaluator for {evaluator_type}")

    @classmethod
    def build_train_loader(cls, cfg):
        # Register datasets
        try:
            _root = os.getenv("DETECTRON2_DATASETS", "datasets")
            register_all_ytvis_2019_incremental(_root, cfg, train=True)
            register_all_ytvis_2021_incremental(_root, cfg, train=True)
            register_all_ovis_incremental(_root, cfg, train=True)
            register_all_coco_video_incremental(_root, cfg)
        except:
            pass

        mappers = []
        for d_i, dataset_name in enumerate(cfg.DATASETS.TRAIN):
            is_tgt = (d_i == len(cfg.DATASETS.TRAIN) - 1)
            if dataset_name.startswith('coco'):
                mappers.append(CocoClipDatasetMapper(cfg, is_train=True, is_tgt=is_tgt, src_dataset_name=dataset_name))
            elif dataset_name.startswith('ytvis') or dataset_name.startswith('ovis'):
                mappers.append(YTVISDatasetMapper(cfg, is_train=True, is_tgt=is_tgt, src_dataset_name=dataset_name))

        if len(mappers) == 1:
            return build_detection_train_loader(cfg, mapper=mappers[0], dataset_name=cfg.DATASETS.TRAIN[0])
        else:
            loaders = [build_detection_train_loader(cfg, mapper=m, dataset_name=d)
                      for m, d in zip(mappers, cfg.DATASETS.TRAIN)]
            return build_combined_loader(cfg, loaders, cfg.DATASETS.DATASET_RATIO)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        if dataset_name.startswith('coco'):
            mapper = CocoClipDatasetMapper(cfg, is_train=False)
        elif dataset_name.startswith('ytvis') or dataset_name.startswith('ovis'):
            mapper = YTVISDatasetMapper(cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Evaluate and save results with task ID.
        """
        results = super().test(cfg, model, evaluators)

        # Save results with task ID for continual learning evaluation
        if comm.is_main_process():
            import shutil
            inference_dir = os.path.join(cfg.OUTPUT_DIR, "inference")
            results_file = os.path.join(inference_dir, "results.json")

            if os.path.exists(results_file):
                # Copy to parent directory with task ID
                output_base = os.path.dirname(cfg.OUTPUT_DIR)
                dst_file = os.path.join(output_base, f"results_{cfg.CONT.TASK}.json")
                shutil.copy(results_file, dst_file)
                logger.info(f"Saved results to {dst_file}")

            # Save metrics to text file
            metrics_file = os.path.join(cfg.OUTPUT_DIR, f"metrics_task{cfg.CONT.TASK}.txt")
            with open(metrics_file, 'w') as f:
                f.write(f"Task {cfg.CONT.TASK} Evaluation Results\n")
                f.write("="*50 + "\n")
                for key, val in results.items():
                    f.write(f"{key}: {val}\n")
            logger.info(f"Saved metrics to {metrics_file}")

        return results


def setup(args):
    """Create configs and perform basic setups."""
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_vita_config(cfg)
    add_continual_config(cfg)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Update number of classes based on current task
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = cfg.CONT.BASE_CLS + cfg.CONT.TASK * cfg.CONT.INC_CLS

    # Setup output directory
    task_name = f"{cfg.DATASETS.TRAIN[0]}_{cfg.CONT.BASE_CLS}-{cfg.CONT.INC_CLS}"
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, task_name, f"task{cfg.CONT.TASK}")

    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="vita")
    return cfg


def main(args):
    cfg = setup(args)

    # Load previous task weights for incremental tasks
    if cfg.CONT.TASK > 0:
        if cfg.CONT.WEIGHTS is None:
            prev_task_dir = cfg.OUTPUT_DIR.replace(f"task{cfg.CONT.TASK}", f"task{cfg.CONT.TASK-1}")
            cfg.defrost()
            cfg.MODEL.WEIGHTS = os.path.join(prev_task_dir, "model_final.pth")
            cfg.freeze()
        logger.info(f"Loading weights from previous task: {cfg.MODEL.WEIGHTS}")

    if args.eval_only:
        model = ContinualTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = ContinualTrainer.test(cfg, model)
        return res

    trainer = ContinualTrainer(cfg)
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
