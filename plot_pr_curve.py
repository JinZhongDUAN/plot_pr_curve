import os
import mmcv
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.signal import savgol_filter
from mmcv import Config
from mmdet.datasets import build_dataset

def getPRArray(config_file, result_file, metric="bbox"):
    cfg = Config.fromfile(config_file)
    # turn on error_analysis mode of dataset
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    # build dataset
    dataset = build_dataset(cfg.data.test)
    # load result file in pkl format
    pkl_results = mmcv.load(result_file)
    # convert pkl file (list[list | tuple | ndarray]) to json
    json_results, _ = dataset.format_results(pkl_results)
    # initialize COCO instance
    coco = COCO(annotation_file=cfg.data.test.ann_file)
    coco_gt = coco
    coco_dt = coco_gt.loadRes(json_results[metric])
    # initialize COCOeval instance
    coco_eval = COCOeval(coco_gt, coco_dt, metric)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    # extract eval data
    precisions = coco_eval.eval["precision"]
    return precisions

def smooth_curve(y, window_size, order):
    return savgol_filter(y, window_size, order)

def compute_average_precision(precisions):
    # Calculate mean precision across IoU thresholds for each recall level
    pr_array = np.mean(precisions[0, :, :, 0, 2], axis=1)
    # for i in range(1, 6):
    #     pr_array += np.mean(precisions[0, :, i, 0, 2], axis=0)
    # mean_precisions = pr_array / 6
    return pr_array

def calculate_f1(precision, recall):
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def calculate_intersection(precision, recall):
    # Find the index where the precision and recall are closest to each other
    idx = np.argmin(np.abs(precision - recall))
    return precision[idx], recall[idx]

def PR(configs, results, labels, out):
    iou_thresholds = np.arange(0.5, 0.95, 0.05)
    x = np.arange(0.0, 1.01, 0.01)
    # Plot PR curves for each model
    threshold = 0.55

    for config, result, label in zip(configs, results, labels):
        precisions = getPRArray(config, result)
        
        mean_precisions = compute_average_precision(precisions)
        smoothed_curve = smooth_curve(mean_precisions, window_size=11, order=3)
        plt.plot(x, smoothed_curve, label=label)

        # Calculate intersection with the diagonal line (slope 1)
        intersection_point = calculate_intersection(smoothed_curve, x)
        precision, recall = intersection_point

        # Calculate F1 score
        f1 = calculate_f1(precision, recall)
        print(f"{label} - Intersection Point (P, R): ({precision}, {recall}), F1: {f1}")

    plt.xticks(np.arange(0, 1.01, 0.1))
    plt.yticks(np.arange(0, 1.01, 0.1))
    font = {'family': 'Noto Sans CJK JP',
             'size': 14,
             }
    plt.xlabel("召回率", font)
    plt.ylabel("精确率", font)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.legend(loc='lower left')
    plt.savefig(out, bbox_inches="tight")
    plt.close()

# model = ['SSD512', 'Faster RCNN', 'FCOS', 'Sparse RCNN', 'RetinaNet', 'Cascade RCNN', 'YOLOX-s', 'Reasoning-RCNN', 'PKR-Net']
def main():
    parser = ArgumentParser(description='COCO PR Curve Tool')
    parser.add_argument('--configs', nargs='+', help='list of error_analysis config file paths')
    parser.add_argument('--results', nargs='+', help='list of prediction paths where error_analysis pkl result')
    parser.add_argument('--labels', nargs='+', default=['SSD512', 'Faster RCNN',
        'FCOS', 'Sparse RCNN', 'RetinaNet', 'Cascade RCNN', 'YOLOX-s', 'Reasoning RCNN', 'CANet', 'PKR-Net'],
                        help='list of model labels')
    parser.add_argument('--out_dir', default='work_dirs/results/test.png', help='dir to save analyze result images')

    args = parser.parse_args()
    PR(args.configs, args.results, args.labels, args.out_dir)

if __name__ == '__main__':
    main()
