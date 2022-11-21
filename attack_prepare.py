import os
import numpy as np
import torch
import argparse
from model_init import model_initial
from generate_attack import target_attack, untarget_attack

parser = argparse.ArgumentParser(description='StyleFool_attack_prepare')
parser.add_argument('--model', type=str, default='C3D', choices=['C3D', 'I3D'], help='the attacked model')
parser.add_argument('--dataset', type=str, default='UCF101', choices=['UCF101', 'HMDB51'], help='the dataset')
parser.add_argument('--gpu',  type=int, default=0, help='use which gpu')
parser.add_argument('--dataset-npy-path', type=str, default='dataset/UCF-101_npy/', help='the path of dataset in npy forms')
parser.add_argument('--target', action='store_true', help='target attack or untarget attack (default)')
parser.add_argument('--styled-npy-path', type=str, default='styled_npy/', help='styled_npy_path')
parser.add_argument('--output-npy-path', type=str, default='output_npy/', help='output_adversarial_npy_path')

def main():
    args = parser.parse_args()
    if args.gpu >= 0:
        device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    target = args.target
    mod = args.model
    dataset = args.dataset
    npy_path = args.dataset_npy_path
    styled_npy_path = args.styled_npy_path
    output_npy_path = args.output_npy_path
    if not os.path.exists(output_npy_path):
        os.mkdir(output_npy_path)
    assert dataset == 'UCF101' and mod == 'C3D'
    model = model_initial(mod, dataset, device)

    class_info = dict()
    with open("classInd.txt", "r") as f:
        lines = f.readlines()
    for line in lines:
        label, name = line.strip("\n").split(" ")
        class_info[name] = int(label) - 1
    if target:
        styled_npy_path = styled_npy_path + "v_target/"
        output_npy_path = output_npy_path + "adv_target/"
        if not os.path.exists(output_npy_path):
            os.mkdir(output_npy_path)
        target_attack(model, class_info, npy_path, styled_npy_path, output_npy_path, args.gpu)
    else:
        styled_npy_path = styled_npy_path + "v_untarget/"
        output_npy_path = output_npy_path + "adv_untarget/"
        if not os.path.exists(output_npy_path):
            os.mkdir(output_npy_path)
        untarget_attack(class_info, styled_npy_path, output_npy_path, args.gpu)


if __name__ == '__main__':
    main()
