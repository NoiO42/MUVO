import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from datetime import datetime
from muvo.data.dataset import DataModule

anomaly_scores = []

def calculate_temporal_difference(prior_pred, prior_batch):
    sum = np.zeros(prior_pred.shape)
    for batch_entry in prior_batch:
        sum += np.abs(prior_pred - batch_entry)
    diff = sum * (1 / len(prior_batch))
    return (1/3) * (diff[:,:,0] + diff[:,:,1] + diff[:,:,2])

def link_losses(perceptual_loss, mse_loss, ssim_loss, l1_loss, temp_diff, runcounter, eval_dir_name):
    l1_loss_scaled = (np.abs(l1_loss[:,:,0]) + np.abs(l1_loss[:,:,1])
                           + np.abs(l1_loss[:,:,2])) * (1/3)
    mse_loss_scaled = (np.abs(mse_loss[:,:,0]) + np.abs(mse_loss[:,:,1]) + np.abs(mse_loss[:,:,2])) * (1/3)
    ssim_loss_scaled = (np.abs(ssim_loss[:,:,0]) + np.abs(ssim_loss[:,:,1]) + np.abs(ssim_loss[:,:,2])) * (1/3)
    np.save('/path_to_eval_file/' + eval_dir_name + '/temp/' + str(DataModule.initial_index) + "_" + str(runcounter), temp_diff)
    linked_output = DataModule.w_per * perceptual_loss + DataModule.w_mse * mse_loss_scaled + DataModule.w_ssim * ssim_loss_scaled + DataModule.w_abs * l1_loss_scaled + DataModule.w_temp * temp_diff
    anomaly_scores.append(linked_output)
    return Image.fromarray((linked_output * 255).astype(np.uint8))

def calculate_masked_loss(mask, anomaly_map):
    return np.sum(np.multiply(mask, anomaly_map))/np.count_nonzero(mask)


def iterate_through_instances(instance_seg_map, anomaly_map):
    map_with_anomalies = np.zeros(instance_seg_map.shape)
    for current_instance in np.unique(np.vstack(instance_seg_map), axis=0):
        mask = np.all(instance_seg_map == current_instance, axis=-1).astype(int)
        masked_loss = calculate_masked_loss(mask, anomaly_map)
        anomaly_scores.append(masked_loss)
        indices = np.where(np.all(instance_seg_map == current_instance, axis=-1))
        map_with_anomalies[indices] = masked_loss
    # max_anomaly_score = np.max(map_with_anomalies)
    # output = np.multiply(anomaly_map, (map_with_anomalies[:,:,0] == max_anomaly_score).astype(int))
    return map_with_anomalies

