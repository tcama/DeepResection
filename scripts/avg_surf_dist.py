import numpy as np
from surface_distance import metrics
import nibabel as nib
import os
import glob
from skimage.measure import label

gt_list = glob.glob('data/nii_tle/*/*resection_mask.nii.gz')
combined_list = [(gt, gt.replace('resection', 'predicted')) for gt in gt_list]

gt_to_pred = []
pred_to_gt = []

def clean_mask(mask):
    mask_data_int = mask.astype(np.uint8)
    labels =label(mask_data_int, connectivity = 2)
    object_labels = list(np.unique(labels))
    object_labels.remove(0)

    # get the continuous object that has the largest volume
    MAX_OBJ = 0
    MAX_OBJ_VOL = 0
    for obj_lab in object_labels:
        vol = np.sum(labels == obj_lab)
        if vol > MAX_OBJ_VOL:
            MAX_OBJ_VOL = vol
            MAX_OBJ = obj_lab

    # remove every other object from the mask
    if MAX_OBJ != 0:
        object_labels.remove(MAX_OBJ)
    for obj_lab in object_labels:
        idx = (labels == obj_lab)
        mask[idx] = 0

for gt_dir, pred_dir in combined_list:
    gt = nib.load(gt_dir)
    gt_data = gt.get_fdata()
    pred = nib.load(pred_dir)
    pred_data = pred.get_fdata()
    clean_mask(gt_data)
    clean_mask(pred_data)
    gt_data = gt_data.astype(bool)
    pred_data = pred_data.astype(bool)
    header = gt.header
    voxel_dims = header.get_zooms()
    sds = metrics.compute_surface_distances(gt_data, pred_data, voxel_dims)
    avgs = metrics.compute_average_surface_distance(sds)
    avg1 = avgs[0]
    avg2 = avgs[1]
    gt_to_pred.append(avg1)
    pred_to_gt.append(avg2)

gt_to_pred = np.array(gt_to_pred)
pred_to_gt = np.array(pred_to_gt)

np.save('gt_to_pred.npy', gt_to_pred)
np.save('pred_to_gt.npy', pred_to_gt)