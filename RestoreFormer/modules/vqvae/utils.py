from torchvision.ops import roi_align
import torch

def get_roi_regions(gt, output, loc_left_eyes, loc_right_eyes, loc_mouths,
                    face_ratio=1, eye_out_size=80, mouth_out_size=120):
    # hard code
    eye_out_size *= face_ratio
    mouth_out_size *= face_ratio

    eye_out_size = int(eye_out_size)
    mouth_out_size = int(mouth_out_size)

    rois_eyes = []
    rois_mouths = []
    for b in range(loc_left_eyes.size(0)):  # loop for batch size
        # left eye and right eye
        img_inds = loc_left_eyes.new_full((2, 1), b)
        bbox = torch.stack([loc_left_eyes[b, :], loc_right_eyes[b, :]], dim=0)  # shape: (2, 4)
        rois = torch.cat([img_inds, bbox], dim=-1)  # shape: (2, 5)
        rois_eyes.append(rois)
        # mouse
        img_inds = loc_left_eyes.new_full((1, 1), b)
        rois = torch.cat([img_inds, loc_mouths[b:b + 1, :]], dim=-1)  # shape: (1, 5)
        rois_mouths.append(rois)

    rois_eyes = torch.cat(rois_eyes, 0)
    rois_mouths = torch.cat(rois_mouths, 0)

    # real images
    all_eyes = roi_align(gt, boxes=rois_eyes, output_size=eye_out_size) * face_ratio
    left_eyes_gt = all_eyes[0::2, :, :, :]
    right_eyes_gt = all_eyes[1::2, :, :, :]
    mouths_gt = roi_align(gt, boxes=rois_mouths, output_size=mouth_out_size) * face_ratio
    # output
    all_eyes = roi_align(output, boxes=rois_eyes, output_size=eye_out_size) * face_ratio
    left_eyes = all_eyes[0::2, :, :, :]
    right_eyes = all_eyes[1::2, :, :, :]
    mouths = roi_align(output, boxes=rois_mouths, output_size=mouth_out_size) * face_ratio

    return {'left_eyes_gt': left_eyes_gt, 'right_eyes_gt': right_eyes_gt, 'mouths_gt': mouths_gt, 
            'left_eyes': left_eyes, 'right_eyes': right_eyes, 'mouths': mouths}
