import numpy as np
import cv2
import os
from sklearn.metrics import confusion_matrix,accuracy_score,ConfusionMatrixDisplay


def calculate_recall(pred_mask,gt_mask,rate=0.03):
    h,w = pred_mask.shape
    D = (int)(h * rate)
    fp = 0
    tp = 0
    fn = 0
    for i in range(h):
        for j in range(w):
            if gt_mask[i][j] > 0:
                fn += 1
            if pred_mask[i][j] > 0:
                fp += 1
            if pred_mask[i][j] > 0:
                if pred_mask[i][j] == gt_mask[i][j]:
                    tp += 1
                    continue
                flag = 0
                for k in range(-D,D):
                    for m in range(-D,D):
                        if k * k + m * m > D:
                            continue
                        if 0 < i + k < 512 and 0 < j + m < 512:
                            if gt_mask[i + k][j + m] > 0:
                                tp += 1
                                flag = 1
                                break
                    if flag == 1:
                        break
    fp -= tp
    fn -= tp
    if fn < 0:
        fp -= fn
        fn = 0
    tn = 512 * 512 - fn - tp - fp
    return tn,fp,fn,tp


def calculate_metrics(pred_mask,gt_mask):
    tn,fp,fn,tp = calculate_recall(pred_mask,gt_mask)
    print(tn,fp,fn,tp,tp / (tp + fn + 1e-10))
    print((tp + fn) / (512 * 512))

    metrics = {
        "IoU": tp / (tp + fp + fn + 1e-10),
        "Dice": 2 * tp / (2 * tp + fp + fn + 1e-10),
        "Accuracy": (tp + tn) / (tp + tn + fp + fn),
        "Precision": tp / (tp + fp + 1e-10),
        "Recall": tp / (tp + fn + 1e-10),
        "Specificity": tn / (tn + fp + 1e-10),
        "FPR": fp / (fp + tn)
    }

    return metrics


pred_dir = ["../data/ground/1/"]
gt_dir = ["../data/ground/1-ground/"]

all_metrics = []

for i in range(len(pred_dir)):
    pred_files = [j for j in os.listdir(pred_dir[i]) if j.endswith('.png')]
    gt_files = [j for j in os.listdir(gt_dir[i]) if j.endswith('.png')]

    for j in range(len(pred_files)):
        pred_path = pred_dir[i] + pred_files[j]
        gt_path = gt_dir[i] + gt_files[j]

        print(pred_path,gt_path)
        pred = cv2.imread(pred_path,cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gt_path,cv2.IMREAD_GRAYSCALE)

        h,w = gt.shape
        zeroNum = 0
        for a in range(h):
            for b in range(w):
                if gt[a,b] != 0:
                    zeroNum += 1
        if zeroNum == 0:
            metrics = {
                "IoU": 0,
                "Dice": 0,
                "Accuracy": 1,
                "Precision": 1,
                "Recall": 1,
                "Specificity": 1,
                "FPR": 0
            }
            all_metrics.append(metrics)
            continue
        assert pred.shape == gt.shape,f"{pred_files[j]} 尺寸不匹配"

        metrics = calculate_metrics(pred,gt)
        all_metrics.append(metrics)

precision = 0
precision_max = -1
precision_min = 2

recall = 0
recall_max = -1
recall_min = 2

fpr = 0
fpr_max = -1
fpr_min = 2

for i in all_metrics:
    if (i['Precision'] > precision_max):
        precision_max = i['Precision']
    if (i['Precision'] < precision_min):
        precision_min = i['Precision']
    if (i['Recall'] > recall_max):
        recall_max = i['Recall']
    if (i['Recall'] < recall_min):
        recall_min = i['Recall']
    if (i['FPR'] > fpr_max):
        fpr_max = i['FPR']
    if (i['FPR'] < fpr_min):
        fpr_min = i['FPR']

    precision += i['Precision']
    recall += i['Recall']
    fpr += i['FPR']

precision = (precision - (precision_min + precision_max)) / (len(all_metrics) - 2)
recall = (recall - (recall_min + recall_max)) / (len(all_metrics) - 2)
fpr = (fpr - (fpr_max +fpr_min)) / (len(all_metrics) - 2)
print(precision,recall,fpr)