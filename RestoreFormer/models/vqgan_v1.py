import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from main import instantiate_from_config

from RestoreFormer.modules.vqvae.utils import get_roi_regions

class RestoreFormerModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="lq",
                 colorize_nlabels=None,
                 monitor=None,
                 special_params_lr_scale=1.0,
                 comp_params_lr_scale=1.0,
                 schedule_step=[80000, 200000]
                 ):
        super().__init__()
        self.image_key = image_key
        self.vqvae = instantiate_from_config(ddconfig)

        lossconfig['params']['distill_param']=ddconfig['params']
        self.loss = instantiate_from_config(lossconfig)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        if lossconfig['params']['comp_weight'] or lossconfig['params']['comp_style_weight']:
            self.use_facial_disc = True
        else:
            self.use_facial_disc = False

        self.fix_decoder = ddconfig['params']['fix_decoder']
        
        self.disc_start = lossconfig['params']['disc_start']
        self.special_params_lr_scale = special_params_lr_scale
        self.comp_params_lr_scale = comp_params_lr_scale
        self.schedule_step = schedule_step

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())

        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]

        state_dict = self.state_dict()
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

        self.load_state_dict(state_dict, strict=True)
        print(f"Restored from {path}")

    def forward(self, input):
        dec, diff, info, hs = self.vqvae(input)
        return dec, diff, info, hs

    def training_step(self, batch, batch_idx, optimizer_idx):
        
        x = batch[self.image_key]
        xrec, qloss, info, hs = self(x)

        if self.image_key != 'gt':
            x = batch['gt']

        if self.use_facial_disc:
            loc_left_eyes = batch['loc_left_eye']
            loc_right_eyes = batch['loc_right_eye']
            loc_mouths = batch['loc_mouth']
            face_ratio = xrec.shape[-1] / 512
            components = get_roi_regions(x, xrec, loc_left_eyes, loc_right_eyes, loc_mouths, face_ratio)
        else:
            components = None

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, components, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, components, optimizer_idx, self.global_step,
                                            last_layer=None, split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

        
        if self.disc_start <= self.global_step:

            # left eye
            if optimizer_idx == 2:
                # discriminator
                disc_left_loss, log_dict_disc = self.loss(qloss, x, xrec, components, optimizer_idx, self.global_step,
                                                last_layer=None, split="train")
                self.log("train/disc_left_loss", disc_left_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
                return disc_left_loss

            # right eye
            if optimizer_idx == 3:
                # discriminator
                disc_right_loss, log_dict_disc = self.loss(qloss, x, xrec, components, optimizer_idx, self.global_step,
                                                last_layer=None, split="train")
                self.log("train/disc_right_loss", disc_right_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
                return disc_right_loss

            # mouth
            if optimizer_idx == 4:
                # discriminator
                disc_mouth_loss, log_dict_disc = self.loss(qloss, x, xrec, components, optimizer_idx, self.global_step,
                                                last_layer=None, split="train")
                self.log("train/disc_mouth_loss", disc_mouth_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
                return disc_mouth_loss

    def validation_step(self, batch, batch_idx):
        x = batch[self.image_key]
        xrec, qloss, info, hs = self(x)

        if self.image_key != 'gt':
            x = batch['gt']

        aeloss, log_dict_ae = self.loss(qloss, x, xrec, None, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, None, 1, self.global_step,
                                            last_layer=None, split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)

        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate

        normal_params = []
        special_params = []
        for name, param in self.vqvae.named_parameters():
            if not param.requires_grad:
                continue
            if 'decoder' in name and 'attn' in name:
                special_params.append(param)
            else:
                normal_params.append(param)
        # print('special_params', special_params)
        opt_ae_params = [{'params': normal_params, 'lr': lr},
                         {'params': special_params, 'lr': lr*self.special_params_lr_scale}]
        opt_ae = torch.optim.Adam(opt_ae_params, betas=(0.5, 0.9))


        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))

        optimizations = [opt_ae, opt_disc]

        s0 = torch.optim.lr_scheduler.MultiStepLR(opt_ae, milestones=self.schedule_step, gamma=0.1, verbose=True)
        s1 = torch.optim.lr_scheduler.MultiStepLR(opt_disc, milestones=self.schedule_step, gamma=0.1, verbose=True)
        schedules = [s0, s1]

        if self.use_facial_disc:
            opt_l = torch.optim.Adam(self.loss.net_d_left_eye.parameters(),
                                     lr=lr*self.comp_params_lr_scale, betas=(0.9, 0.99))
            opt_r = torch.optim.Adam(self.loss.net_d_right_eye.parameters(),
                                     lr=lr*self.comp_params_lr_scale, betas=(0.9, 0.99))
            opt_m = torch.optim.Adam(self.loss.net_d_mouth.parameters(),
                                     lr=lr*self.comp_params_lr_scale, betas=(0.9, 0.99))
            optimizations += [opt_l, opt_r, opt_m]
            
            s2 = torch.optim.lr_scheduler.MultiStepLR(opt_l, milestones=self.schedule_step, gamma=0.1, verbose=True)
            s3 = torch.optim.lr_scheduler.MultiStepLR(opt_r, milestones=self.schedule_step, gamma=0.1, verbose=True)
            s4 = torch.optim.lr_scheduler.MultiStepLR(opt_m, milestones=self.schedule_step, gamma=0.1, verbose=True)
            schedules += [s2, s3, s4]

        return optimizations, schedules

    def get_last_layer(self):
        if self.fix_decoder:
            return self.vqvae.quant_conv.weight
        return self.vqvae.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = batch[self.image_key]
        x = x.to(self.device)
        xrec, _, _, _ = self(x)
        log["inputs"] = x
        log["reconstructions"] = xrec

        if self.image_key != 'gt':
            x = batch['gt']
            log["gt"] = x
        return log