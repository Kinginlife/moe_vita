"""
OVIS dataset support with incremental learning for VITA.
Adapted from HVPL hvpl/data/datasets/ovis.py
"""

# COCO to OVIS category mapping
COCO_TO_OVIS = {
    1:1, 2:21, 3:6, 4:21, 5:28, 7:17, 8:29, 9:34, 17:14, 18:8, 19:18,
    21:15, 22:32, 23:20, 24:30, 25:22, 35:33, 36:33, 41:5, 42:27, 43:40
}

OVIS_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "Person"},
    {"color": [0, 82, 0], "isthing": 1, "id": 2, "name": "Bird"},
    {"color": [119, 11, 32], "isthing": 1, "id": 3, "name": "Cat"},
    {"color": [165, 42, 42], "isthing": 1, "id": 4, "name": "Dog"},
    {"color": [134, 134, 103], "isthing": 1, "id": 5, "name": "Horse"},
    {"color": [0, 0, 142], "isthing": 1, "id": 6, "name": "Sheep"},
    {"color": [255, 109, 65], "isthing": 1, "id": 7, "name": "Cow"},
    {"color": [0, 226, 252], "isthing": 1, "id": 8, "name": "Elephant"},
    {"color": [5, 121, 0], "isthing": 1, "id": 9, "name": "Bear"},
    {"color": [0, 60, 100], "isthing": 1, "id": 10, "name": "Zebra"},
    {"color": [250, 170, 30], "isthing": 1, "id": 11, "name": "Giraffe"},
    {"color": [100, 170, 30], "isthing": 1, "id": 12, "name": "Poultry"},
    {"color": [179, 0, 194], "isthing": 1, "id": 13, "name": "Giant_panda"},
    {"color": [255, 77, 255], "isthing": 1, "id": 14, "name": "Lizard"},
    {"color": [120, 166, 157], "isthing": 1, "id": 15, "name": "Parrot"},
    {"color": [73, 77, 174], "isthing": 1, "id": 16, "name": "Monkey"},
    {"color": [0, 80, 100], "isthing": 1, "id": 17, "name": "Rabbit"},
    {"color": [182, 182, 255], "isthing": 1, "id": 18, "name": "Tiger"},
    {"color": [0, 143, 149], "isthing": 1, "id": 19, "name": "Fish"},
    {"color": [174, 57, 255], "isthing": 1, "id": 20, "name": "Turtle"},
    {"color": [0, 0, 230], "isthing": 1, "id": 21, "name": "Bicycle"},
    {"color": [72, 0, 118], "isthing": 1, "id": 22, "name": "Motorcycle"},
    {"color": [255, 179, 240], "isthing": 1, "id": 23, "name": "Airplane"},
    {"color": [0, 125, 92], "isthing": 1, "id": 24, "name": "Boat"},
    {"color": [209, 0, 151], "isthing": 1, "id": 25, "name": "Vehical"},
]


def _get_ovis_instances_meta():
    thing_ids = [k["id"] for k in OVIS_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in OVIS_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 25, len(thing_ids)
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in OVIS_CATEGORIES if k["isthing"] == 1]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
