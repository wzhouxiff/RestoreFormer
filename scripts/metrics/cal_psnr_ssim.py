import os, sys
import argparse
import cv2
import numpy as np
import glob
import pdb
import tqdm
import torch

from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim

root_path = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir))
sys.path.append(root_path)
sys.path.append(os.path.join(root_path, 'RestoreFormer/modules/losses'))

from lpips import LPIPS

def calculate_psnr_ssim_lpips_folder():
    parser = argparse.ArgumentParser()

    parser.add_argument('folder', type=str, help='Path to the folder')
    parser.add_argument('--gt_folder', type=str, help='Path to the GT')
    parser.add_argument('--save_name', type=str, default='niqe', help='File name for saving results')
    parser.add_argument('--need_post', type=int, default=0, help='0: the name of image does not include 00, 1: otherwise')

    args = parser.parse_args()

    fout = open(args.save_name, 'w')
    fout.write('NAME\tPSNR\tSSIM\tLPIPS\n')

    H, W = 512, 512

    gt_names = glob.glob(os.path.join(args.gt_folder, '*'))
    gt_names.sort()

    perceptual_loss = LPIPS().eval().cuda()

    mean_psnr = 0.
    mean_ssim = 0.
    mean_lpips = 0.
    mean_norm_lpips = 0.

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

        cur_psnr = calculate_psnr(img, gt, 0)
        cur_ssim = calculate_ssim(img, gt, 0)

        # lpips:
        img = img.astype(np.float32) / 255.
        img = torch.FloatTensor(img).cuda()
        img = img.permute(2,0,1)
        img = img.unsqueeze(0)

        gt = gt.astype(np.float32) / 255.
        gt = torch.FloatTensor(gt).cuda()
        gt = gt.permute(2,0,1)
        gt = gt.unsqueeze(0)

        cur_lpips = perceptual_loss(img, gt)
        cur_lpips = cur_lpips[0].item()

        img = (img - 0.5) / 0.5
        gt = (gt - 0.5) / 0.5

        norm_lpips = perceptual_loss(img, gt)
        norm_lpips = norm_lpips[0].item()

        # print(cur_psnr, cur_ssim, cur_lpips, norm_lpips)

        fout.write(gt_name + '\t' + str(cur_psnr) + '\t' + str(cur_ssim) + '\t' + str(cur_lpips) + '\t' + str(norm_lpips) + '\n')

        mean_psnr += cur_psnr
        mean_ssim += cur_ssim
        mean_lpips += cur_lpips
        mean_norm_lpips += norm_lpips

    mean_psnr /= float(len(gt_names))
    mean_ssim /= float(len(gt_names))
    mean_lpips /= float(len(gt_names))
    mean_norm_lpips /= float(len(gt_names))

    fout.write(str(mean_psnr) + '\t' + str(mean_ssim) + '\t' + str(mean_lpips) + '\t' + str(mean_norm_lpips) + '\n')
    fout.close()

    print('psnr, ssim, lpips, norm_lpips:', mean_psnr, mean_ssim, mean_lpips, mean_norm_lpips)

if __name__ == '__main__':
    calculate_psnr_ssim_lpips_folder()