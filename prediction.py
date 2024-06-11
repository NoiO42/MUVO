import os
import socket
import time
from tqdm import tqdm
from muvo.config import _C
from muvo.models.transition import RSSM
import torchvision.transforms.functional as tvf

import torch
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np

from muvo.models.mile import Mile
from muvo.config import get_parser, get_cfg
from muvo.data.dataset import DataModule
from muvo.trainer import WorldModelTrainer
from muvo.losses import SpatialRegressionLoss
from muvo.anomaly_detection.linkage_module import link_losses, iterate_through_instances, calculate_temporal_difference
from muvo.data.dataset import DataModule
from lightning.pytorch.callbacks import ModelSummary

from clearml import Task, Dataset, Model

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage import img_as_float


from datetime import datetime

PATH_TO_ANOVOX = ''
PATH_TO_PRETRAINED_MUVO_MODEL = 'no_Voxel_epoch.12-step.100000.ckpt'
MUVO_OUTPUT_PATH = ''

def crop(array : np.ndarray):
    for i in range(len(array)):
        for j in range(len(array[0])):
            for z in range(len(array[0][0])):
                if(array[i][j][z] > 1):
                    array[i][j][z] = 1.0
                if(array[i][j][z] < 0):
                    array[i][j][z] = 0.0

def main():
    runcounter = 0
    args = get_parser().parse_args()
    eval_dir_name = args.dir
    DataModule.w_abs = args.abs
    DataModule.w_mse = args.mse
    DataModule.w_ssim = args.ssim
    DataModule.w_per = args.per
    DataModule.w_temp = args.temp
    DataModule.initial_index = args.start
    DataModule.threshold = args.threshold
    cfg = get_cfg(args)

    task = Task.init(project_name=cfg.CML_PROJECT, task_name=('eval_' + eval_dir_name), task_type=cfg.CML_TYPE, tags=cfg.TAG)
    task.connect(cfg)
    cml_logger = task.get_logger()
    
    dataset_root = PATH_TO_ANOVOX + '/Final_Output_2024_02_22-13_38/'

    data = DataModule(cfg, dataset_root=dataset_root)
    data.setup()
    input_model = PATH_TO_PRETRAINED_MUVO_MODEL
    model = WorldModelTrainer(cfg.convert_to_dict(), pretrained_path=input_model)

    save_dir = os.path.join(
        cfg.LOG_DIR, time.strftime('%d%B%Yat%H:%M:%S%Z') + '_' + socket.gethostname() + '_' + cfg.TAG
    )
    writer = SummaryWriter(log_dir=save_dir)

    dataloader = data.test_dataloader()[2]

    pbar = tqdm(total=len(dataloader),  desc='Prediction')
    model.cuda()

    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()

    n_prediction_samples = model.cfg.PREDICTION.N_SAMPLES

    for i, batch in enumerate(dataloader):
        output_imagines = []
        ims = []
        batch = {key: value.cuda() for key, value in batch.items()}
        with torch.no_grad():
            batch = model.preprocess(batch)
            batch_rf = {key: value[:, :model.rf] for key, value in batch.items()}
            output, state_dict = model.model.forward(batch_rf, deployment=False)

            state_imagine = {'hidden_state': state_dict['posterior']['hidden_state'][:, -1],
                             'sample': state_dict['posterior']['sample'][:, -1],
                             'throttle_brake': batch['throttle_brake'][:, model.rf:],
                             'steering': batch['steering'][:, model.rf:]}
            for _ in range(n_prediction_samples):
                output_imagine = model.model.imagine(state_imagine, predict_action=False, future_horizon=model.fh)
                output_imagines.append(output_imagine)
                voxel_im = torch.where(torch.argmax(output_imagine['voxel_1'].detach(), dim=-4).cpu() != 0)
                voxel_im = torch.stack(voxel_im).transpose(0, 1).numpy()
                ims.append({'rgb_re': output_imagine['rgb_1'].detach().cpu().numpy(),
                            'pcd_re': output_imagine['lidar_reconstruction_1'].detach().cpu().numpy(),
                            'voxel_re': voxel_im,
                            })
                
            gt = {'rgb_label': batch['rgb_label_1'].cpu().numpy(),
                  'throttle_brake': batch['throttle_brake'].cpu().numpy(),
                  'steering': batch['steering'].cpu().numpy(),
                  'pcd_label': batch['range_view_label_1'].cpu().numpy(),
                  }
            for i in range(len(Mile.prior_rgb_output)):
                difference_img = np.abs(batch['rgb_label_1'].cpu().numpy()[0][i] - Mile.prior_rgb_output[i])
                crop(difference_img)
                r_data = difference_img[0]
                g_data = difference_img[1]
                b_data = difference_img[2]
                output_image = np.zeros(difference_img[0].shape)
                for j in range(len(output_image)):
                    for x in range(len(output_image[0])):
                        if(r_data[j][x] > 0.42 or g_data[j][x] > 0.42 or b_data[j][x] > 0.42):
                            output_image[j][x] = 1.0
            rgb_loss = SpatialRegressionLoss(norm=1)
            training_loss = rgb_loss.forward(prediction=Mile.prior_raw_output, target=batch['rgb_label_1'])
            for i in range(len(batch['rgb_label_1'].cpu().numpy()[0])):
                r_data = (batch['rgb_label_1'].cpu().numpy()[0][i][0])
                g_data = (batch['rgb_label_1'].cpu().numpy()[0][i][1])
                b_data = (batch['rgb_label_1'].cpu().numpy()[0][i][2])
                img = np.dstack([r_data, g_data, b_data])
                crop(img)
            r_data = (batch['rgb_label_1'].cpu().numpy()[0][-1][0])
            g_data = (batch['rgb_label_1'].cpu().numpy()[0][-1][1])
            b_data = (batch['rgb_label_1'].cpu().numpy()[0][-1][2])
            img = np.dstack([r_data, g_data, b_data])
            crop(img)
            np.save(MUVO_OUTPUT_PATH + eval_dir_name + '/ground_truth/' + str(DataModule.initial_index) + "_" + str(runcounter), img)
            left, top, right, bottom = _C.IMAGE.CROP
            height = bottom - top
            width = right - left
            sem_seg_img = tvf.crop(Image.fromarray((batch['instance_segmentation_map'].cpu().numpy()[0][-1][:,:,0:3] * 255).astype(np.uint8))
                                   , top, left, height, width)
            np.save(MUVO_OUTPUT_PATH + eval_dir_name + '/instance/' + str(DataModule.initial_index) + "_" + str(runcounter), np.array(sem_seg_img))
            sem_seg_img.save(MUVO_OUTPUT_PATH + eval_dir_name + '/instance_img/' + str(DataModule.initial_index) + "_" + str(runcounter) + '.png')
            semantic_map = tvf.crop(Image.fromarray((batch['semantic_segmentation_map'].cpu().numpy()[0][-1] * 255).astype(np.uint8)), top, left, height, width)
            semantic_map.save(MUVO_OUTPUT_PATH + eval_dir_name + '/semantic/' +
                              str(DataModule.initial_index) + "_" + str(runcounter) + '.png')
            r_data = (Mile.prior_rgb_output[-1][0])
            g_data = (Mile.prior_rgb_output[-1][1])
            b_data = (Mile.prior_rgb_output[-1][2])
            prior_img = np.dstack([r_data, g_data, b_data])
            crop(prior_img)
            plt.imsave(MUVO_OUTPUT_PATH + eval_dir_name + '/prior/' + str(DataModule.initial_index) + "_" + str(runcounter) + '.png', prior_img)
            print("Spatial regression loss: " + str(training_loss.cpu().numpy()))
            print("MSE: " + str(np.mean((img - prior_img) ** 2)))
            ssim_value, ssim_img = ssim(img, prior_img,data_range=np.max([np.max(img), np.max(prior_img)])
                                         - np.min([np.min(img), np.min(prior_img)]), channel_axis=2, full=True)
            print("SSIM: " + str(ssim_value))
            temporal_diff_img = calculate_temporal_difference(prior_img, Mile.prior_batch)
            np.save('/path_to_eval_file/' + eval_dir_name + '/temp/' + str(DataModule.initial_index) + "_" + str(runcounter), temporal_diff_img)
            runcounter = runcounter + 1
            re = {'rgb_re': output['rgb_1'].detach().cpu().numpy(),
                  'pcd_re': output['lidar_reconstruction_1'].detach().cpu().numpy(),
                  }
            upload_data = {'gt': gt, 're': re, 'ims': ims}
            task.upload_artifact(f'data_{i}', np.array(upload_data))
        pbar.update(1)
            

if __name__ == '__main__':
    main()
