import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from RestoreFormer.modules.losses.lpips import LPIPS
from RestoreFormer.modules.discriminator.model import NLayerDiscriminator, weights_init
from RestoreFormer.modules.vqvae.facial_component_discriminator import FacialComponentDiscriminator
from basicsr.losses.losses import GANLoss, L1Loss
from RestoreFormer.modules.vqvae.arcface_arch import ResNetArcFace


class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


class VQLPIPSWithDiscriminatorWithCompWithIdentity(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, 
                 disc_ndf=64, disc_loss="hinge", comp_weight=0.0, comp_style_weight=0.0, 
                 identity_weight=0.0, comp_disc_loss='vanilla', lpips_style_weight=0.0,
                 identity_model_path=None, **ignore_kwargs):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS(style_weight=lpips_style_weight).eval()
        self.perceptual_weight = perceptual_weight

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm,
                                                 ndf=disc_ndf
                                                 ).apply(weights_init)
        if comp_weight > 0:
            self.net_d_left_eye = FacialComponentDiscriminator()
            self.net_d_right_eye = FacialComponentDiscriminator()
            self.net_d_mouth = FacialComponentDiscriminator()
            print(f'Use components discrimination')

            self.cri_component = GANLoss(gan_type=comp_disc_loss, 
                                         real_label_val=1.0, 
                                         fake_label_val=0.0, 
                                         loss_weight=comp_weight)

            if comp_style_weight > 0.:
                self.cri_style = L1Loss(loss_weight=comp_style_weight, reduction='mean')

        if identity_weight > 0:
            self.identity = ResNetArcFace(block = 'IRBlock', 
                                          layers = [2, 2, 2, 2],
                                          use_se = False)
            print(f'Use identity loss')
            if identity_model_path is not None:
                sd = torch.load(identity_model_path, map_location="cpu")
                for k, v in deepcopy(sd).items():
                    if k.startswith('module.'):
                        sd[k[7:]] = v
                        sd.pop(k)
                self.identity.load_state_dict(sd, strict=True)

            for param in self.identity.parameters():
                param.requires_grad = False

            self.cri_identity = L1Loss(loss_weight=identity_weight, reduction='mean')


        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminatorWithCompWithIdentity running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.comp_weight = comp_weight
        self.comp_style_weight = comp_style_weight
        self.identity_weight = identity_weight
        self.lpips_style_weight = lpips_style_weight

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram

    def gray_resize_for_identity(self, out, size=128):
        out_gray = (0.2989 * out[:, 0, :, :] + 0.5870 * out[:, 1, :, :] + 0.1140 * out[:, 2, :, :])
        out_gray = out_gray.unsqueeze(1)
        out_gray = F.interpolate(out_gray, (size, size), mode='bilinear', align_corners=False)
        return out_gray

    def forward(self, codebook_loss, gts, reconstructions, components, optimizer_idx,
                global_step, last_layer=None, split="train"):

        # now the GAN part
        if optimizer_idx == 0:
            rec_loss = (torch.abs(gts.contiguous() - reconstructions.contiguous())) * self.pixel_weight
            if self.perceptual_weight > 0:
                p_loss, p_style_loss = self.perceptual_loss(gts.contiguous(), reconstructions.contiguous())
                rec_loss = rec_loss + self.perceptual_weight * p_loss
            else:
                p_loss = torch.tensor([0.0])
                p_style_loss = torch.tensor([0.0])

            nll_loss = rec_loss
            #nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
            nll_loss = torch.mean(nll_loss)

        
            # generator update
            
            logits_fake = self.discriminator(reconstructions.contiguous())
            g_loss = -torch.mean(logits_fake)

            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            
            loss = nll_loss + d_weight * disc_factor * g_loss + self.codebook_weight * codebook_loss.mean() + p_style_loss

            log = {
                   "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/p_loss".format(split): p_loss.detach().mean(),
                   "{}/p_style_loss".format(split): p_style_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }

            if self.comp_weight > 0. and components is not None and self.discriminator_iter_start < global_step:
                fake_left_eye, fake_left_eye_feats = self.net_d_left_eye(components['left_eyes'], return_feats=True)
                comp_g_loss = self.cri_component(fake_left_eye, True, is_disc=False)
                loss = loss + comp_g_loss 
                log["{}/g_left_loss".format(split)] = comp_g_loss.detach()

                fake_right_eye, fake_right_eye_feats = self.net_d_right_eye(components['right_eyes'], return_feats=True)
                comp_g_loss = self.cri_component(fake_right_eye, True, is_disc=False)
                loss = loss + comp_g_loss 
                log["{}/g_right_loss".format(split)] = comp_g_loss.detach()

                fake_mouth, fake_mouth_feats = self.net_d_mouth(components['mouths'], return_feats=True)
                comp_g_loss = self.cri_component(fake_mouth, True, is_disc=False)
                loss = loss + comp_g_loss 
                log["{}/g_mouth_loss".format(split)] = comp_g_loss.detach()

                if self.comp_style_weight > 0.:
                    _, real_left_eye_feats = self.net_d_left_eye(components['left_eyes_gt'], return_feats=True)
                    _, real_right_eye_feats = self.net_d_right_eye(components['right_eyes_gt'], return_feats=True)
                    _, real_mouth_feats = self.net_d_mouth(components['mouths_gt'], return_feats=True)

                    def _comp_style(feat, feat_gt, criterion):
                        return criterion(self._gram_mat(feat[0]), self._gram_mat(
                            feat_gt[0].detach())) * 0.5 + criterion(self._gram_mat(
                            feat[1]), self._gram_mat(feat_gt[1].detach()))

                    comp_style_loss = 0.
                    comp_style_loss = comp_style_loss + _comp_style(fake_left_eye_feats, real_left_eye_feats, self.cri_style)
                    comp_style_loss = comp_style_loss + _comp_style(fake_right_eye_feats, real_right_eye_feats, self.cri_style)
                    comp_style_loss = comp_style_loss + _comp_style(fake_mouth_feats, real_mouth_feats, self.cri_style)
                    loss = loss + comp_style_loss 
                    log["{}/comp_style_loss".format(split)] = comp_style_loss.detach()

            if self.identity_weight > 0. and self.discriminator_iter_start < global_step:
                self.identity.eval()
                out_gray = self.gray_resize_for_identity(reconstructions)
                gt_gray = self.gray_resize_for_identity(gts)
                
                identity_gt = self.identity(gt_gray).detach()
                identity_out = self.identity(out_gray)

                identity_loss = self.cri_identity(identity_out, identity_gt)
                loss = loss + identity_loss 
                log["{}/identity_loss".format(split)] = identity_loss.detach()

            log["{}/total_loss".format(split)] = loss.clone().detach().mean()

            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            
            logits_real = self.discriminator(gts.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log

        # left eye
        if optimizer_idx == 2:
            # third pass for discriminator update
            disc_factor = adopt_weight(1.0, global_step, threshold=self.discriminator_iter_start)
            fake_d_pred, _ = self.net_d_left_eye(components['left_eyes'].detach())
            real_d_pred, _ = self.net_d_left_eye(components['left_eyes_gt'])
            d_loss = self.cri_component(real_d_pred, True, is_disc=True) + self.cri_component(fake_d_pred, False, is_disc=True)

            log = {"{}/d_left_loss".format(split): d_loss.clone().detach().mean()}
            return d_loss, log

        # right eye
        if optimizer_idx == 3:
            # forth pass for discriminator update
            fake_d_pred, _ = self.net_d_right_eye(components['right_eyes'].detach())
            real_d_pred, _ = self.net_d_right_eye(components['right_eyes_gt'])
            d_loss = self.cri_component(real_d_pred, True, is_disc=True) + self.cri_component(fake_d_pred, False, is_disc=True)

            log = {"{}/d_right_loss".format(split): d_loss.clone().detach().mean()}
            return d_loss, log

        # mouth
        if optimizer_idx == 4:
            # fifth pass for discriminator update
            fake_d_pred, _ = self.net_d_mouth(components['mouths'].detach())
            real_d_pred, _ = self.net_d_mouth(components['mouths_gt'])
            d_loss = self.cri_component(real_d_pred, True, is_disc=True) + self.cri_component(fake_d_pred, False, is_disc=True)

            log = {"{}/d_mouth_loss".format(split): d_loss.clone().detach().mean()}
            return d_loss, log
