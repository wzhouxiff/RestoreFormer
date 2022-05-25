import argparse, os, sys, glob, math, time
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import pdb

sys.path.append(os.getcwd())

from main import instantiate_from_config, DataModuleFromConfig
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import trange, tqdm

import cv2
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from torchvision.transforms.functional import normalize

from basicsr.utils import img2tensor, imwrite, tensor2img


def restoration(model,
                face_helper,
                img_path,
                save_root,
                has_aligned=False,
                only_center_face=True,
                suffix=None,
                paste_back=False):
    # read image
    img_name = os.path.basename(img_path)
    # print(f'Processing {img_name} ...')
    basename, _ = os.path.splitext(img_name)
    input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    face_helper.clean_all()

    if has_aligned:
        input_img = cv2.resize(input_img, (512, 512))
        face_helper.cropped_faces = [input_img]
    else:
        face_helper.read_image(input_img)
        # get face landmarks for each face
        face_helper.get_face_landmarks_5(only_center_face=only_center_face, pad_blur=False)
        # align and warp each face
        save_crop_path = os.path.join(save_root, 'cropped_faces', img_name)
        face_helper.align_warp_face(save_crop_path)

    # face restoration
    for idx, cropped_face in enumerate(face_helper.cropped_faces):
        # prepare data
        cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        cropped_face_t = cropped_face_t.unsqueeze(0).to('cuda')

        try:
            with torch.no_grad():
                output = model(cropped_face_t)
                restored_face = tensor2img(output[0].squeeze(0), rgb2bgr=True, min_max=(-1, 1))
        except RuntimeError as error:
            print(f'\tFailed inference for GFPGAN: {error}.')
            restored_face = cropped_face

        restored_face = restored_face.astype('uint8')
        face_helper.add_restored_face(restored_face)

        if suffix is not None:
            save_face_name = f'{basename}_{idx:02d}_{suffix}.png'
        else:
            save_face_name = f'{basename}_{idx:02d}.png'
        save_restore_path = os.path.join(save_root, 'restored_faces', save_face_name)
        imwrite(restored_face, save_restore_path)


    if not has_aligned and paste_back:
        face_helper.get_inverse_affine(None)
        save_restore_path = os.path.join(save_root, 'restored_imgs', img_name)
        # paste each restored face to the input image
        face_helper.paste_faces_to_input_image(save_restore_path)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-c",
        "--config",
        nargs="?",
        metavar="single_config.yaml",
        help="path to single config. If specified, base configs will be ignored "
        "(except for the last one if left unspecified).",
        const=True,
        default="",
    )
    parser.add_argument(
        "--ignore_base_data",
        action="store_true",
        help="Ignore data specification from base configs. Useful if you want "
        "to specify a custom datasets on the command line.",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        type=str,
        help="Where to write outputs to.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="Sample from among top-k predictions.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    parser.add_argument('--upscale_factor', type=int, default=1)
    parser.add_argument('--test_path', type=str, default='inputs/whole_imgs')
    parser.add_argument('--suffix', type=str, default=None, help='Suffix of the restored faces')
    parser.add_argument('--only_center_face', action='store_true')
    parser.add_argument('--aligned', action='store_true')
    parser.add_argument('--paste_back', action='store_true')

    return parser


def load_model_from_config(config, sd, gpu=True, eval_mode=True):
    if "ckpt_path" in config.params:
        print("Deleting the restore-ckpt path from the config...")
        config.params.ckpt_path = None
    if "downsample_cond_size" in config.params:
        print("Deleting downsample-cond-size from the config and setting factor=0.5 instead...")
        config.params.downsample_cond_size = -1
        config.params["downsample_cond_factor"] = 0.5
    try:
        if "ckpt_path" in config.params.first_stage_config.params:
            config.params.first_stage_config.params.ckpt_path = None
            print("Deleting the first-stage restore-ckpt path from the config...")
        if "ckpt_path" in config.params.cond_stage_config.params:
            config.params.cond_stage_config.params.ckpt_path = None
            print("Deleting the cond-stage restore-ckpt path from the config...")
    except:
        pass

    model = instantiate_from_config(config)
    if sd is not None:
        keys = list(sd.keys())

        state_dict = model.state_dict()
        require_keys = state_dict.keys()
        keys = sd.keys()
        un_pretrained_keys = []
        for k in require_keys:
            if k not in keys: 
                # miss 'vqvae.'
                if k[6:] in keys:
                    state_dict[k] = sd[k[6:]]
                else:
                    un_pretrained_keys.append(k)
            else:
                state_dict[k] = sd[k]

        # print(f'*************************************************')
        # print(f"Layers without pretraining: {un_pretrained_keys}")
        # print(f'*************************************************')

        model.load_state_dict(state_dict, strict=True)

    if gpu:
        model.cuda()
    if eval_mode:
        model.eval()
    return {"model": model}


def load_model_and_dset(config, ckpt, gpu, eval_mode):

    # now load the specified checkpoint
    if ckpt:
        pl_sd = torch.load(ckpt, map_location="cpu")
    else:
        pl_sd = {"state_dict": None}

    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"],
                                   gpu=gpu,
                                   eval_mode=eval_mode)["model"]
    return model

if __name__ == "__main__":
    sys.path.append(os.getcwd())

    parser = get_parser()

    opt, unknown = parser.parse_known_args()

    ckpt = None
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            try:
                idx = len(paths)-paths[::-1].index("logs")+1
            except ValueError:
                idx = -2 # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
        print(f"logdir:{logdir}")
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*-project.yaml")))
        opt.base = base_configs+opt.base

    if opt.config:
        if type(opt.config) == str:
            if not os.path.exists(opt.config):
                raise ValueError("Cannot find {}".format(opt.config))
            if os.path.isfile(opt.config):
                opt.base = [opt.config]
            else:
                opt.base = sorted(glob.glob(os.path.join(opt.config, "*-project.yaml")))
        else:
            opt.base = [opt.base[-1]]

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    if opt.ignore_base_data:
        for config in configs:
            if hasattr(config, "data"): del config["data"]
    config = OmegaConf.merge(*configs, cli)
    
    print(config)
    gpu = True
    eval_mode = True
    show_config = False
    if show_config:
        print(OmegaConf.to_container(config))

    model = load_model_and_dset(config, ckpt, gpu, eval_mode)
    
    outdir = opt.outdir
    os.makedirs(outdir, exist_ok=True)
    print("Writing samples to ", outdir)

    # initialize face helper
    face_helper = FaceRestoreHelper(
        opt.upscale_factor, face_size=512, crop_ratio=(1, 1), det_model='retinaface_resnet50', save_ext='png')

    img_list = sorted(glob.glob(os.path.join(opt.test_path, '*')))

    print('Results are in the <{}> folder.'.format(outdir))
    
    for img_path in tqdm(img_list):
        restoration(
                model,
                face_helper,
                img_path,
                outdir,
                has_aligned=opt.aligned,
                only_center_face=opt.only_center_face,
                suffix=opt.suffix,
                paste_back=opt.paste_back)

    print('Test number: ', len(img_list))
    print('Results are in the <{}> folder.'.format(outdir))
