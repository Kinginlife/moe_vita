"""
Continual learning evaluation for VITA.
Adapted from HVPL dataset_eval.py
"""
import numpy as np
import os
import json
from tqdm import tqdm


def compute_forgetting_metrics(result_dir, num_tasks, base_classes, inc_classes,
                                annFile, forgetting_type='class'):
    """
    Compute forgetting metrics for continual learning.

    Args:
        result_dir: Directory containing result_{task_id}.json files
        num_tasks: Total number of tasks
        base_classes: Number of base classes
        inc_classes: Number of incremental classes per task
        annFile: Path to annotation file for evaluation
        forgetting_type: 'class' or 'task' level forgetting

    Returns:
        dict: Forgetting metrics (FAP, FAR1, etc.)
    """
    from vita.data.ytvis_eval import YTVISEvaluator

    num_classes = base_classes + (num_tasks - 1) * inc_classes

    if forgetting_type == 'task':
        metrix = np.zeros((12, num_tasks, num_tasks))
        max_metric = np.zeros((12, num_tasks))
        count_task = np.zeros((12, num_tasks))
    else:
        metrix = np.zeros((12, num_classes, num_tasks))
        max_metric = np.zeros((12, num_classes))
        count_task = np.zeros((12, num_classes))

    task_pbar = tqdm(range(num_tasks), desc="Evaluating Tasks")

    for task_id in task_pbar:
        classes = base_classes + task_id * inc_classes
        result_file = os.path.join(result_dir, f'results_{task_id}.json')

        if not os.path.exists(result_file):
            print(f"Warning: {result_file} not found, skipping...")
            continue

        task_pbar.set_description(f"Task {task_id} [{classes} Classes]")

        if forgetting_type == 'class':
            cls_pbar = tqdm(range(classes), desc="  Evaluating Classes", leave=False)
            for cls in cls_pbar:
                if task_id == 0 and cls < base_classes:
                    count_task[:, cls] = num_tasks - task_id - 1
                elif task_id > 0 and cls >= (classes - inc_classes) and cls < classes:
                    count_task[:, cls] = num_tasks - task_id - 1

                # Evaluate per-class performance
                # This requires custom evaluator implementation
                # Placeholder for actual evaluation logic
                pass

    # Compute forgetting
    forgetting_results = {}

    if forgetting_type == 'class':
        for idx in range(metrix.shape[0]):
            forgetting_metric = metrix[idx, :, :]
            forgetting = max_metric[idx, :-inc_classes] - forgetting_metric[:-inc_classes, -1] + 1e-8
            forgetting[forgetting < 0] = 0
            forgetting = forgetting / (max_metric[idx, :-inc_classes] + 1e-8)
            forgetting = forgetting / count_task[idx, :-inc_classes]
            forgetting_val = np.sum(forgetting) / (num_classes - inc_classes)
            forgetting_results[f'metric_{idx}'] = forgetting_val

    return forgetting_results


def save_forgetting_results(result_dir, forgetting_results, forgetting_type='class'):
    """Save forgetting results to CSV file."""
    output_file = os.path.join(result_dir, f'{forgetting_type}_forgetting.csv')

    with open(output_file, 'w') as f:
        for key, value in forgetting_results.items():
            f.write(f"{key}: {value}\n")

    print(f"Forgetting results saved to {output_file}")


class ContinualLearningEvaluator:
    """Evaluator for continual learning VIS."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.task_id = cfg.CONT.TASK
        self.base_classes = cfg.CONT.BASE_CLS
        self.inc_classes = cfg.CONT.INC_CLS
        self.current_classes = self.base_classes + self.task_id * self.inc_classes

    def evaluate_task(self, predictions, ground_truth):
        """Evaluate a single task."""
        # Placeholder for task evaluation
        # Should compute AP, AR metrics per class
        pass

    def compute_forgetting(self, all_results):
        """Compute forgetting metrics across all tasks."""
        # Placeholder for forgetting computation
        pass
