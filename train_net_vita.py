try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except ImportError:
    pass

import copy
import itertools
import logging
import os

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
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
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# MaskFormer
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
from detectron2.data import DatasetCatalog, MetadataCatalog

from vita.data.datasets.builtin import register_all_coco_video, register_all_ytvis_2019, register_all_ytvis_2021, register_all_ovis

from vita.data.datasets.builtin import register_all_ytvis_2019_val, register_all_ytvis_2021_val, register_all_ovis_val
logger = logging.getLogger("detectron2")


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to VITA with continual learning and MoE support.
    """

    @classmethod
    def build_model(cls, cfg):
        model = super().build_model(cfg)

        # Handle MoE expert management for incremental tasks
        if cfg.MOE.ENABLED:
            try:
                decoder = model.sem_seg_head.predictor
                # Get all MoE layers
                moe_layers = [layer for layer in decoder.transformer_ffn_layers
                             if hasattr(layer, 'moe')]

                for moe_layer_wrapper in moe_layers:
                    moe_layer = moe_layer_wrapper.moe
                    current_num_experts = len(moe_layer.experts)
                    expected_num_experts = cfg.CONT.TASK + 1  # Task 0 -> 1 expert, Task 1 -> 2 experts, etc.

                    # Add new expert if needed
                    if current_num_experts < expected_num_experts:
                        init_from = cfg.CONT.TASK - 1 if cfg.MOE.INIT_FROM_PREVIOUS and cfg.CONT.TASK > 0 else None
                        moe_layer.add_expert(
                            cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
                            cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD,
                            dropout=0.0,
                            init_from=init_from
                        )
                        logger.info(f"Task {cfg.CONT.TASK}: Added expert {current_num_experts} (total: {expected_num_experts})")

                    # Freeze old experts (exclude current task expert)
                    if cfg.MOE.FREEZE_OLD_EXPERTS and cfg.CONT.TASK > 0:
                        old_expert_ids = list(range(cfg.CONT.TASK))
                        moe_layer.freeze_experts(old_expert_ids)
                        logger.info(f"Frozen old experts: {old_expert_ids}")
            except AttributeError as e:
                logger.error(f"MoE layer not found in model: {e}")
                raise

        return model

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            os.makedirs(output_folder, exist_ok=True)
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        elif evaluator_type == "ytvis":
            evaluator_list.append(YTVISEvaluator(dataset_name, cfg, True, output_folder))

        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        else:
            raise NotImplementedError

    @classmethod
    def build_train_loader(cls, cfg):
        #print(model)
        try:  #使用detectron2使用数据集用DatasetCatalog.get(dataset_name)获取数据,所以要先把数据集注册到DatasetCatalog
            _root_data = os.getenv("DETECTRON2_DATASETS", "datasets")
            register_all_ovis(_root_data, cfg, train=True)
            register_all_ytvis_2019(_root_data, cfg, train=True)
            register_all_ytvis_2021(_root_data, cfg, train=True)
            register_all_coco_video(_root_data, cfg)
        except:
            pass

        mappers = []
        for d_i, dataset_name in enumerate(cfg.DATASETS.TRAIN):
            print(dataset_name)
            if dataset_name.startswith('coco'):
                mappers.append(
                    CocoClipDatasetMapper(
                        cfg, is_train=True, is_tgt=(d_i==len(cfg.DATASETS.TRAIN)-1), src_dataset_name=dataset_name
                    )
                )
            elif dataset_name.startswith('ytvis') or dataset_name.startswith('ovis'):
                mappers.append(
                    YTVISDatasetMapper(cfg, is_train=True, is_tgt=(d_i==len(cfg.DATASETS.TRAIN)-1), src_dataset_name=dataset_name)
                )
            else:
                raise NotImplementedError
        assert len(mappers) > 0, "No dataset is chosen!"

        if len(mappers) == 1:
            mapper = mappers[0]
            return build_detection_train_loader(cfg, mapper=mapper, dataset_name=cfg.DATASETS.TRAIN[0])
        else:

            loaders = []
            for mapper, dataset_name in zip(mappers, cfg.DATASETS.TRAIN):
                dataset = DatasetCatalog.get(dataset_name)
                if len(dataset) == 0:
                    print(dataset_name, 'is empty')
                else:
                    loader = build_detection_train_loader(cfg, mapper=mapper, dataset_name=dataset_name)
                    loaders.append(loader)

            if len(loaders) > 1:
                #loaders = [
                    #build_detection_train_loader(cfg, mapper=mapper, dataset_name=dataset_name)
                    #for mapper, dataset_name in zip(mappers, cfg.DATASETS.TRAIN)
                #]
                combined_data_loader = build_combined_loader(cfg, loaders, cfg.DATASETS.DATASET_RATIO)
                return combined_data_loader

            else:
                return loader

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):

        try:
            _root_data = os.getenv("DETECTRON2_DATASETS", "datasets")
            #register_all_ovis(_root_data, cfg, train=False)
            #register_all_ytvis_2019(_root_data, cfg, train=False)
            #register_all_ytvis_2021(_root_data, cfg, train=False)
            #register_all_coco_video(_root_data, cfg)
            register_all_ytvis_2019_val(_root_data, cfg, train=False)
            register_all_ytvis_2021_val(_root_data, cfg, train=False)
            register_all_ovis_val(_root_data, cfg, train=False)
        
        except:
            pass

        dataset_name = cfg.DATASETS.TEST[0]
        if dataset_name.startswith('coco'):
            mapper = CocoClipDatasetMapper(cfg, is_train=False)
        elif dataset_name.startswith('ytvis') or dataset_name.startswith('ovis'):
            mapper = YTVISDatasetMapper(cfg, is_train=False)
        
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)
    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Evaluate the given model with continual learning result saving.
        """
        from torch.cuda.amp import autocast
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            with autocast():
                results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        # Save continual learning results
        if comm.is_main_process() and hasattr(cfg, 'CONT'):
            import shutil
            inference_dir = os.path.join(cfg.OUTPUT_DIR, "inference")
            results_file = os.path.join(inference_dir, "results.json")
            if os.path.exists(results_file):
                output_base = os.path.dirname(cfg.OUTPUT_DIR)
                dst_file = os.path.join(output_base, f"results_{cfg.CONT.TASK}.json")
                shutil.copy(results_file, dst_file)
                logger.info(f"Saved results to {dst_file}")

            metrics_file = os.path.join(cfg.OUTPUT_DIR, f"metrics_task{cfg.CONT.TASK}.txt")
            with open(metrics_file, 'w') as f:
                for key, val in results.items():
                    f.write(f"{key}: {val}\n")
            logger.info(f"Saved metrics to {metrics_file}")

        if len(results) == 1:
            results = list(results.values())[0]
        return results

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_vita_config(cfg)
    add_continual_config(cfg)  # Add continual learning config
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            raise NotImplementedError
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    # Set task_id for MoE if enabled
    if cfg.MOE.ENABLED and hasattr(cfg, 'CONT'):
        trainer.model.set_task_id(cfg.CONT.TASK)
        logger.info(f"Set model task_id to {cfg.CONT.TASK}")

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
