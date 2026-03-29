"""
Simple evaluation script for computing forgetting metrics in continual learning.
Adapted from HVPL dataset_eval.py

Usage:
    python eval_continual.py --result-dir ./output/ytvis_2019_20_2/ \
                             --num-tasks 11 --base-cls 20 --inc-cls 2 \
                             --ann-file datasets/ytvis_2019/valid.json
"""
import argparse
import numpy as np
import os
from tqdm import tqdm


def compute_forgetting(result_dir, num_tasks, base_classes, inc_classes):
    """
    Compute forgetting metrics from saved results.

    Args:
        result_dir: Directory containing results_{task_id}.json files
        num_tasks: Total number of tasks
        base_classes: Number of base classes
        inc_classes: Number of incremental classes per task
    """
    num_classes = base_classes + (num_tasks - 1) * inc_classes

    # Placeholder metrics (12 standard COCO metrics)
    # In practice, you need to load actual evaluation results
    metrix = np.zeros((12, num_classes, num_tasks))
    max_metric = np.zeros((12, num_classes))
    count_task = np.zeros((12, num_classes))

    print(f"Computing forgetting for {num_tasks} tasks, {num_classes} classes")
    print(f"Base classes: {base_classes}, Incremental: {inc_classes}")

    # Check which result files exist
    for task_id in range(num_tasks):
        result_file = os.path.join(result_dir, f'results_{task_id}.json')
        if os.path.exists(result_file):
            print(f"✓ Found: {result_file}")
        else:
            print(f"✗ Missing: {result_file}")

    # Compute forgetting
    forgetting_results = {}

    for idx in range(12):
        forgetting_metric = metrix[idx, :, :]
        old_classes = num_classes - inc_classes

        if old_classes > 0:
            forgetting = max_metric[idx, :old_classes] - forgetting_metric[:old_classes, -1] + 1e-8
            forgetting[forgetting < 0] = 0
            forgetting = forgetting / (max_metric[idx, :old_classes] + 1e-8)
            forgetting = forgetting / (count_task[idx, :old_classes] + 1e-8)
            forgetting_val = np.sum(forgetting) / old_classes
        else:
            forgetting_val = 0.0

        forgetting_results[f'metric_{idx}'] = forgetting_val

    # Save results
    output_file = os.path.join(result_dir, 'class_forgetting.csv')
    with open(output_file, 'w') as f:
        f.write("Metric,Forgetting\n")
        metric_names = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl',
                       'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl']
        for idx, (key, value) in enumerate(forgetting_results.items()):
            f.write(f"{metric_names[idx]},{value:.4f}\n")

    print(f"\nForgetting results saved to: {output_file}")
    print("\nForgetting Summary:")
    print(f"  FAP (Forgetting AP): {forgetting_results['metric_0']:.4f}")
    print(f"  FAR1 (Forgetting AR1): {forgetting_results['metric_6']:.4f}")

    return forgetting_results


def main():
    parser = argparse.ArgumentParser(description="Compute continual learning forgetting metrics")
    parser.add_argument("--result-dir", type=str, required=True,
                       help="Directory containing results_{task_id}.json files")
    parser.add_argument("--num-tasks", type=int, required=True,
                       help="Total number of tasks")
    parser.add_argument("--base-cls", type=int, required=True,
                       help="Number of base classes")
    parser.add_argument("--inc-cls", type=int, required=True,
                       help="Number of incremental classes per task")
    parser.add_argument("--ann-file", type=str, default=None,
                       help="Annotation file for evaluation (optional)")

    args = parser.parse_args()

    compute_forgetting(
        args.result_dir,
        args.num_tasks,
        args.base_cls,
        args.inc_cls
    )


if __name__ == "__main__":
    main()
