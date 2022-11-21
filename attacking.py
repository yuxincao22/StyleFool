import logging
import os
import sys
import numpy as np
import torch
from attack.video_attack import targeted_video_attack, untargeted_video_attack
from model_wrapper.vid_model_top_k import C3D_K_Model
from utils.args_attack import video_attack_args_parse
from models import C3D

def main():
    args = video_attack_args_parse()
    # parameters setting
    untargeted = args.untargeted
    rank_transform = not args.no_rank_transform
    sigma = args.sigma
    sample_per_draw = args.sample_per_draw
    sub_num_sample = args.sub_num_sample
    gpus = args.gpus

    os.environ["CUDA_VISIBLE_DEVICES"] = ', '.join([str(gpu) for gpu in gpus])
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(message)s')

    model = C3D(num_classes=101, pretrained=False).cuda()
    checkpoint = torch.load(
        'ckpt/C3D-ucf101_epoch-49.pth.tar',
        map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    vid_model = C3D_K_Model(model)

    vid = np.load(args.video).transpose(0, 3, 1, 2)/255
    vid = torch.tensor(vid, dtype=torch.float, device='cuda')
    vid_label = args.label

    if not untargeted:
        target_vid = np.load(args.target_video).transpose(0, 3, 1, 2)/255
        target_vid = torch.tensor(target_vid, dtype=torch.float, device='cuda')
        target_label = args.target_label
        res, iter_num, adv_vid = targeted_video_attack(vid_model, vid, target_vid,
                                                       target_label, rank_transform=rank_transform,
                                                       sub_num_sample=sub_num_sample, sigma=sigma,
                                                       eps=0.05, max_iter=300000,
                                                       sample_per_draw=sample_per_draw)
    else:
        res, iter_num, adv_vid = untargeted_video_attack(vid_model, vid,
                                                         vid_label, rank_transform=rank_transform,
                                                         sub_num_sample=sub_num_sample, sigma=sigma,
                                                         eps=0.05, max_iter=300000,
                                                         sample_per_draw=sample_per_draw)
        top_val, top_idx, _ = vid_model(adv_vid[None, :])
        target_label = top_idx[0][0].item()
    adv_vid = adv_vid.cpu().numpy()
    if res:
        if untargeted:
            logging.info(
                '-----untargeted attack succeed using {} quries-----'.format(iter_num))
        else:
            logging.info(
                '-----{} transfer to {} using {} quries-----'.format(vid_label, target_label, iter_num))
    elif iter_num > 300000:
        logging.info('-----Attack Fails > 3e6-----')
    else:
        logging.info('-----Attack Fails-----')
    np.save(args.adv_save_path, adv_vid)


if __name__ == '__main__':
    main()
