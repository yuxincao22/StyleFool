import os
import numpy as np
import torch


def target_attack(model, class_info, npy_path, styled_npy_path, output_npy_path, gpu=0):
    npy_path_class = sorted(os.listdir(npy_path))
    npy_info = {}
    for subdir in npy_path_class:
        sub_path = npy_path + subdir + "/"
        sub_sub_npy = sorted(os.listdir(sub_path))
        for subsubdir in sub_sub_npy:
            sub_sub_path = sub_path + subsubdir
            npy_info[subsubdir] = sub_sub_path
    styled_npy_path_class = sorted(os.listdir(styled_npy_path))
    styled_info = {}
    styled_all_npy = []
    for subdir in styled_npy_path_class:
        styled_sub_path = styled_npy_path + subdir
        styled_info[subdir] = styled_sub_path
        styled_all_npy.append(subdir)
    with open("target_attack_batch.sh", "w") as ff:
        for i in range(len(styled_all_npy)):
            npy_name = styled_all_npy[i]
            vid_path = styled_info[npy_name]
            video = np.load(vid_path).transpose(0, 3, 1, 2) / 255
            video = torch.tensor(video, dtype=torch.float, device='cuda')
            top_val, top_idx, _ = model(video[None, :])
            vid_label = top_idx[0][0] # the stylized label
            target_vid_name = npy_name.split("stylized_")[1].split("_00")[0] + ".npy"
            target_vid_class_name = target_vid_name.split("_")[1]
            target_label = class_info[target_vid_class_name]
            target_vid_path = npy_info[target_vid_name]
            adv_save_path = output_npy_path + npy_name.split(".")[0] + "_" + target_vid_name.split(".")[0] + ".npy"
            command = "python attacking.py --gpus %s --adv-save-path %s --target-video %s --video %s " \
                      "--label %d --target-label %d --sub_num_sample 4 --sigma 1e-6\n" \
                      % (str(gpu), adv_save_path, target_vid_path, vid_path, vid_label, target_label)
            print(command)
            ff.writelines("echo 'begin'\n")
            ff.writelines(command)
            ff.writelines("echo 'end'\n\n")


def untarget_attack(class_info, styled_npy_path, output_npy_path, gpu=1):
    styled_npy_path_class = sorted(os.listdir(styled_npy_path))
    styled_info = {}
    styled_all_npy = []
    for subdir in styled_npy_path_class:
        styled_sub_path = styled_npy_path + subdir
        styled_info[subdir] = styled_sub_path
        styled_all_npy.append(subdir)
    with open("untarget_attack_batch.sh", "w") as ff:
        for i in range(len(styled_all_npy)):
            npy_name = styled_all_npy[i]
            vid_path = styled_info[npy_name]
            vid_class_name = vid_path.split("/")[-1]
            vid_label = class_info[vid_class_name.split("_")[1]]

            adv_save_path = output_npy_path + npy_name.split(".")[0] + "_untarget" + ".npy"
            command = "python attacking.py --gpus %s --untargeted --sigma 1e-3 --adv-save-path %s --video %s " \
                      "--label %d --sub_num_sample 4\n" \
                      % (str(gpu), adv_save_path, vid_path, vid_label)
            print(command)
            ff.writelines("echo 'begin'\n")
            ff.writelines(command)
            ff.writelines("echo 'end'\n\n")
