# Copyright (c) Facebook, Inc. and its affiliates.
"""
COCO dataset loader with incremental learning support for VITA.
Adapted from HVPL hvpl/data/datasets/Additional_coco_increment.py
"""
import contextlib
import io
import json
import logging
import numpy as np
import os
import pycocotools.mask as mask_util
from fvcore.common.timer import Timer
from PIL import Image

from detectron2.structures import Boxes, BoxMode, PolygonMasks
from detectron2.utils.file_io import PathManager
from detectron2.data import DatasetCatalog, MetadataCatalog

logger = logging.getLogger(__name__)

__all__ = ["load_coco_json_incremental", "register_coco_instances_incremental"]


def load_coco_json_incremental(json_file, image_root, dataset_name=None, extra_annotation_keys=None, cfg=None):
    """
    Load COCO json with incremental learning class filtering.

    Args:
        json_file: Path to COCO json file
        image_root: Directory containing images
        dataset_name: Name of the dataset
        extra_annotation_keys: Additional annotation keys to load
        cfg: Config object containing CONT settings

    Returns:
        list[dict]: Dataset dicts in Detectron2 format
    """
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    # Get COCO to target dataset mapping
    from .ytvis_increment import COCO_TO_YTVIS_2019, COCO_TO_YTVIS_2021
    from .ovis_increment import COCO_TO_OVIS

    if cfg.DATASETS.TRAIN[-1].startswith("ytvis_2019"):
        src2tgt = COCO_TO_YTVIS_2019
    elif cfg.DATASETS.TRAIN[-1].startswith("ytvis_2021"):
        src2tgt = COCO_TO_YTVIS_2021
    elif cfg.DATASETS.TRAIN[-1].startswith("ovis"):
        src2tgt = COCO_TO_OVIS
    else:
        src2tgt = {}

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(coco_api.getCatIds())
        cats = coco_api.loadCats(cat_ids)
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        meta.thing_classes = thing_classes

        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning("Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.")
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map

    img_ids = sorted(coco_api.imgs.keys())
    imgs = coco_api.loadImgs(img_ids)
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]

    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []
    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"] + (extra_annotation_keys or [])
    num_instances_without_valid_segmentation = 0

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            assert anno["image_id"] == image_id
            assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'

            obj = {key: anno[key] for key in ann_keys if key in anno}
            if "bbox" in obj and len(obj["bbox"]) == 0:
                raise ValueError(f"One annotation of image {image_id} contains empty 'bbox' value!")

            segm = anno.get("segmentation", None)
            if segm:
                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                else:
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue
                obj["segmentation"] = segm

            keypts = anno.get("keypoints", None)
            if keypts:
                for idx, v in enumerate(keypts):
                    if idx % 3 != 2:
                        keypts[idx] = v + 0.5
                obj["keypoints"] = keypts

            obj["bbox_mode"] = BoxMode.XYWH_ABS

            if id_map:
                annotation_category_id = obj["category_id"]

                # Incremental learning class filtering
                if src2tgt and annotation_category_id in src2tgt:
                    target_class_id = src2tgt[annotation_category_id] - 1  # Convert to 0-indexed

                    # Filter based on current task
                    if cfg.CONT.TASK == 0:
                        # Base task: only include base classes
                        if target_class_id < cfg.CONT.BASE_CLS:
                            try:
                                obj["category_id"] = id_map[annotation_category_id]
                                objs.append(obj)
                            except KeyError as e:
                                raise KeyError(
                                    f"Encountered category_id={annotation_category_id} "
                                    "but this id does not exist in 'categories' of the json file."
                                ) from e
                    else:
                        # Incremental task: only include current task's new classes
                        task_start = cfg.CONT.BASE_CLS + (cfg.CONT.TASK - 1) * cfg.CONT.INC_CLS
                        task_end = cfg.CONT.BASE_CLS + cfg.CONT.TASK * cfg.CONT.INC_CLS

                        if task_start <= target_class_id < task_end:
                            try:
                                obj["category_id"] = id_map[annotation_category_id]
                                objs.append(obj)
                            except KeyError as e:
                                raise KeyError(
                                    f"Encountered category_id={annotation_category_id} "
                                    "but this id does not exist in 'categories' of the json file."
                                ) from e

        if len(objs) > 0:
            record["annotations"] = objs
            dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out  instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
        )

    logger.info(f"Filtered to {len(dataset_dicts)} images for task {cfg.CONT.TASK}")
    return dataset_dicts


def register_coco_instances_incremental(name, metadata, json_file, image_root, cfg=None):
    """
    Register a COCO dataset with incremental learning support.

    Args:
        name: Dataset name
        metadata: Metadata dict
        json_file: Path to COCO json
        image_root: Image directory
        cfg: Config with CONT settings
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root

    DatasetCatalog.register(
        name,
        lambda: load_coco_json_incremental(json_file, image_root, name, cfg=cfg)
    )

    MetadataCatalog.get(name).set(
        json_file=json_file,
        image_root=image_root,
        evaluator_type="coco",
        **metadata
    )
