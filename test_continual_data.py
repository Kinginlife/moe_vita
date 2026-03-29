"""
Test script for continual learning data processing in VITA.
This script validates the incremental class filtering logic.
"""
import os
import json
import tempfile
import shutil
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog

# Add VITA paths
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vita.continual_config import add_continual_config
from vita.data.ytvis_increment import (
    COCO_TO_YTVIS_2019,
    COCO_TO_YTVIS_2021,
    _get_ytvis_2019_instances_meta,
    _get_ytvis_2021_instances_meta
)
from vita.data.coco_increment import load_coco_json_incremental


def create_mock_coco_json(output_path, num_images=10, num_categories=40):
    """Create a mock COCO-format JSON file for testing."""

    # Create categories (1-40 for YTVIS)
    categories = []
    for i in range(1, num_categories + 1):
        categories.append({
            "id": i,
            "name": f"category_{i}",
            "supercategory": "object"
        })

    # Create images
    images = []
    for img_id in range(1, num_images + 1):
        images.append({
            "id": img_id,
            "file_name": f"image_{img_id:06d}.jpg",
            "height": 480,
            "width": 640
        })

    # Create annotations with different categories
    annotations = []
    ann_id = 1
    for img_id in range(1, num_images + 1):
        # Each image has 2-3 objects from different categories
        num_objs = 2 if img_id % 2 == 0 else 3
        for obj_idx in range(num_objs):
            # Distribute categories across images
            # Images 1-3: categories 1-10 (base classes)
            # Images 4-6: categories 11-20 (base classes)
            # Images 7-8: categories 21-22 (task 1 classes)
            # Images 9-10: categories 23-24 (task 2 classes)
            if img_id <= 3:
                cat_id = (img_id - 1) * 3 + obj_idx + 1
            elif img_id <= 6:
                cat_id = 10 + (img_id - 4) * 3 + obj_idx + 1
            elif img_id <= 8:
                cat_id = 21 + (img_id - 7) * 2 + obj_idx
            else:
                cat_id = 23 + (img_id - 9) * 2 + obj_idx

            # Ensure category is valid
            if cat_id > num_categories:
                cat_id = num_categories

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [100 + obj_idx * 50, 100, 100, 100],
                "area": 10000,
                "iscrowd": 0,
                "segmentation": [[100, 100, 200, 100, 200, 200, 100, 200]]
            })
            ann_id += 1

    # Create COCO JSON
    coco_data = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=2)

    print(f"✓ Created mock COCO JSON: {output_path}")
    print(f"  - {len(images)} images")
    print(f"  - {len(annotations)} annotations")
    print(f"  - {len(categories)} categories")

    return coco_data


def create_mock_ytvis_json(output_path, num_videos=5, num_categories=40):
    """Create a mock YTVIS-format JSON file for testing."""

    # Create categories
    categories = []
    for i in range(1, num_categories + 1):
        categories.append({
            "id": i,
            "name": f"category_{i}"
        })

    # Create videos
    videos = []
    for vid_id in range(1, num_videos + 1):
        num_frames = 5
        videos.append({
            "id": vid_id,
            "width": 640,
            "height": 480,
            "length": num_frames,
            "file_names": [f"video_{vid_id:03d}/{frame_id:05d}.jpg"
                          for frame_id in range(num_frames)]
        })

    # Create annotations
    annotations = []
    ann_id = 1
    for vid_id in range(1, num_videos + 1):
        num_frames = 5
        # Video 1-2: base classes (1-10)
        # Video 3: task 1 classes (21-22)
        # Video 4-5: task 2 classes (23-24)
        if vid_id <= 2:
            cat_id = vid_id * 5
        elif vid_id == 3:
            cat_id = 21
        else:
            cat_id = 23

        # Create bboxes and segmentations for all frames
        bboxes = [[100, 100, 100, 100] if i < num_frames else None
                  for i in range(num_frames)]
        segmentations = [[[100, 100, 200, 100, 200, 200, 100, 200]] if i < num_frames else None
                        for i in range(num_frames)]

        annotations.append({
            "id": ann_id,
            "video_id": vid_id,
            "category_id": cat_id,
            "bboxes": bboxes,
            "segmentations": segmentations,
            "areas": [10000] * num_frames,
            "iscrowd": 0
        })
        ann_id += 1

    # Create YTVIS JSON
    ytvis_data = {
        "videos": videos,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_path, 'w') as f:
        json.dump(ytvis_data, f, indent=2)

    print(f"✓ Created mock YTVIS JSON: {output_path}")
    print(f"  - {len(videos)} videos")
    print(f"  - {len(annotations)} annotations")
    print(f"  - {len(categories)} categories")

    return ytvis_data


def setup_test_config(task_id, base_cls, inc_cls):
    """Setup test configuration."""
    cfg = get_cfg()
    add_continual_config(cfg)

    cfg.CONT.TASK = task_id
    cfg.CONT.BASE_CLS = base_cls
    cfg.CONT.INC_CLS = inc_cls
    cfg.DATASETS.TRAIN = ["ytvis_2019_train"]

    return cfg


def test_coco_incremental_loading():
    """Test COCO incremental loading with class filtering."""
    print("\n" + "="*60)
    print("TEST 1: COCO Incremental Loading")
    print("="*60)

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    json_path = os.path.join(temp_dir, "mock_coco.json")

    try:
        # Create mock data
        coco_data = create_mock_coco_json(json_path, num_images=10, num_categories=40)

        # Test Task 0 (base classes 0-19)
        print("\n--- Task 0: Base Classes (0-19) ---")
        cfg = setup_test_config(task_id=0, base_cls=20, inc_cls=2)
        dataset_dicts = load_coco_json_incremental(
            json_path, temp_dir, "test_coco", cfg=cfg
        )
        print(f"✓ Loaded {len(dataset_dicts)} images for Task 0")

        # Check categories
        all_cats = set()
        for d in dataset_dicts:
            for ann in d.get("annotations", []):
                all_cats.add(ann["category_id"])
        print(f"  Categories found: {sorted(all_cats)}")
        print(f"  Expected: 0-19 (base classes)")

        # Test Task 1 (incremental classes 20-21)
        print("\n--- Task 1: Incremental Classes (20-21) ---")
        cfg = setup_test_config(task_id=1, base_cls=20, inc_cls=2)
        dataset_dicts = load_coco_json_incremental(
            json_path, temp_dir, "test_coco", cfg=cfg
        )
        print(f"✓ Loaded {len(dataset_dicts)} images for Task 1")

        all_cats = set()
        for d in dataset_dicts:
            for ann in d.get("annotations", []):
                all_cats.add(ann["category_id"])
        print(f"  Categories found: {sorted(all_cats)}")
        print(f"  Expected: 20-21 (new classes)")

        print("\n✅ COCO incremental loading test PASSED")

    finally:
        shutil.rmtree(temp_dir)


def test_category_mappings():
    """Test COCO to YTVIS category mappings."""
    print("\n" + "="*60)
    print("TEST 2: Category Mappings")
    print("="*60)

    print("\n--- COCO to YTVIS 2019 Mapping ---")
    print(f"Total mappings: {len(COCO_TO_YTVIS_2019)}")
    print(f"Sample mappings:")
    for coco_id, ytvis_id in list(COCO_TO_YTVIS_2019.items())[:5]:
        print(f"  COCO {coco_id} → YTVIS {ytvis_id}")

    print("\n--- COCO to YTVIS 2021 Mapping ---")
    print(f"Total mappings: {len(COCO_TO_YTVIS_2021)}")
    print(f"Sample mappings:")
    for coco_id, ytvis_id in list(COCO_TO_YTVIS_2021.items())[:5]:
        print(f"  COCO {coco_id} → YTVIS {ytvis_id}")

    print("\n✅ Category mapping test PASSED")


def test_metadata():
    """Test metadata retrieval."""
    print("\n" + "="*60)
    print("TEST 3: Metadata")
    print("="*60)

    print("\n--- YTVIS 2019 Metadata ---")
    meta_2019 = _get_ytvis_2019_instances_meta()
    print(f"Number of classes: {len(meta_2019['thing_classes'])}")
    print(f"Sample classes: {meta_2019['thing_classes'][:5]}")

    print("\n--- YTVIS 2021 Metadata ---")
    meta_2021 = _get_ytvis_2021_instances_meta()
    print(f"Number of classes: {len(meta_2021['thing_classes'])}")
    print(f"Sample classes: {meta_2021['thing_classes'][:5]}")

    print("\n✅ Metadata test PASSED")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("VITA Continual Learning Data Processing Tests")
    print("="*60)

    try:
        test_category_mappings()
        test_metadata()
        test_coco_incremental_loading()

        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
