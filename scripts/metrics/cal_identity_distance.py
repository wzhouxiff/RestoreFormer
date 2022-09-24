import os, sys
import torch
import argparse
import cv2
import numpy as np
import glob
import pdb
import tqdm
from copy import deepcopy
import torch.nn.functional as F
import math


root_path = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir))
sys.path.append(root_path)
sys.path.append(os.path.join(root_path, 'RestoreFormer/modules/losses'))

from RestoreFormer.modules.vqvae.arcface_arch import ResNetArcFace
from basicsr.losses.losses import L1Loss, MSELoss

def cosine_similarity(emb1, emb2):
    return np.arccos(np.dot(emb1, emb2) / ( np.linalg.norm(emb1) * np.linalg.norm(emb2)))


def gray_resize_for_identity(out, size=128):
    out_gray = (0.2989 * out[:, 0, :, :] + 0.5870 * out[:, 1, :, :] + 0.1140 * out[:, 2, :, :])
    out_gray = out_gray.unsqueeze(1)
    out_gray = F.interpolate(out_gray, (size, size), mode='bilinear', align_corners=False)
    return out_gray

def calculate_identity_distance_folder():
    parser = argparse.ArgumentParser()

    parser.add_argument('folder', type=str, help='Path to the folder')
    parser.add_argument('--gt_folder', type=str, help='Path to the GT')
    parser.add_argument('--save_name', type=str, default='niqe', help='File name for saving results')
    parser.add_argument('--need_post', type=int, default=0, help='0: the name of image does not include 00, 1: otherwise')

    args = parser.parse_args()

    fout = open(args.save_name, 'w')

    identity = ResNetArcFace(block = 'IRBlock', 
                                  layers = [2, 2, 2, 2],
                                  use_se = False)
    identity_model_path = 'experiments/pretrained_models/arcface_resnet18.pth'
    
    sd = torch.load(identity_model_path, map_location="cpu")
    for k, v in deepcopy(sd).items():
        if k.startswith('module.'):
            sd[k[7:]] = v
            sd.pop(k)
    identity.load_state_dict(sd, strict=True)
    identity.eval()

    for param in identity.parameters():
        param.requires_grad = False

    identity = identity.cuda()

    gt_names = glob.glob(os.path.join(args.gt_folder, '*'))
    gt_names.sort()
    
    mean_dist = 0.
    for i in tqdm.tqdm(range(len(gt_names))):
        gt_name = gt_names[i].split('/')[-1][:-4]
        if args.need_post:
            img_name = os.path.join(args.folder,gt_name + '_00.png')
        else:
            img_name = os.path.join(args.folder,gt_name + '.png')
        if not os.path.exists(img_name):
            print(img_name, 'does not exist')
            continue

        img = cv2.imread(img_name)
        gt = cv2.imread(gt_names[i])

        img = img.astype(np.float32) / 255.
        img = torch.FloatTensor(img).cuda()
        img = img.permute(2,0,1)
        img = img.unsqueeze(0)

        gt = gt.astype(np.float32) / 255.
        gt = torch.FloatTensor(gt).cuda()
        gt = gt.permute(2,0,1)
        gt = gt.unsqueeze(0)

        out_gray = gray_resize_for_identity(img)
        gt_gray = gray_resize_for_identity(gt)

        with torch.no_grad():
            identity_gt = identity(gt_gray)
            identity_out = identity(out_gray)

        identity_gt = identity_gt.cpu().data.numpy().squeeze()
        identity_out = identity_out.cpu().data.numpy().squeeze()
        identity_loss = cosine_similarity(identity_gt, identity_out)

        fout.write(gt_name + ' ' + str(identity_loss) + '\n')
        mean_dist += identity_loss

    fout.write('Mean: ' + str(mean_dist / len(gt_names)) + '\n')
    fout.close()
    print('mean_dist:', mean_dist / len(gt_names))

if __name__ == '__main__':
    calculate_identity_distance_folder()