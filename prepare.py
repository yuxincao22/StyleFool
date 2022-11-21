import os
import torch
import argparse
from utils.utils import calculate_color, select_style_target, select_style_untarget, calculate_superposition
from utils.preprocess import generate_styles
from model_init import model_initial

parser = argparse.ArgumentParser(description='StyleFool_prepare')
parser.add_argument('--model', type=str, default='C3D', choices=['C3D', 'I3D'], help='the attacked model')
parser.add_argument('--dataset', type=str, default='UCF101', choices=['UCF101', 'HMDB51'], help='the dataset')
parser.add_argument('--color', action='store_false', help='need to calculate the color themes')
parser.add_argument('--superposition', action='store_false', help='need to calculate the target class confidence')
parser.add_argument('--style-folder',  type=str, default='./styles/', help='path to save all the style images')
parser.add_argument('--prepare-save-path',  type=str, default='./styles_main_color_csv/', help='path to save color themes and target class confidence')
parser.add_argument('--gpu',  type=int, default=0, help='use which gpu')
parser.add_argument('--dataset-img-path', type=str, default='dataset/UCF-101_images/', help='the path of dataset in png forms')
parser.add_argument('--generate-styles', action='store_false', help='need to generate styles')
parser.add_argument('--dataset-video-path', type=str, default='dataset/UCF-101/', help='the path of dataset in avi forms')
parser.add_argument('--attack-id', type=list, default=[7154, 34456], help='attack_id')
parser.add_argument('--random-seed', type=int, default=16766, help='random_seed')
parser.add_argument('--target', action='store_true', help='target attack or untarget attack (default)')

def main():
    args = parser.parse_args()
    if args.gpu >= 0:
        device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    dataset_img_path = args.dataset_img_path
    video_path = args.dataset_video_path
    prepare_save_path = args.prepare_save_path
    style_folder = args.style_folder
    attack_id = args.attack_id
    random_seed = args.random_seed
    target = args.target
    mod = args.model
    dataset = args.dataset
    assert dataset == 'UCF101' and mod == 'C3D'
    model = model_initial(mod, dataset, device)

    class_info = dict()
    with open("classInd.txt", "r") as f:
        lines = f.readlines()
    for line in lines:
        label, name = line.strip("\n").split(" ")
        class_info[name] = int(label) - 1

    if not os.path.exists(prepare_save_path):
        os.mkdir(prepare_save_path)
    if not os.path.exists(style_folder):
        os.mkdir(style_folder)

    if args.generate_styles:
        print('generating styles ...')
        generate_styles(dataset_img_path, style_folder)
    if args.color:
        print('calculating color themes ...')
        calculate_color(style_folder, prepare_save_path)
    if args.superposition:
        print('calculating target class confidence ...')
        calculate_superposition(model, class_info, style_folder, prepare_save_path, device, mod, dataset)
    if target:
        select_style_target(style_folder, video_path, class_info, attack_id, random_seed)
    else:
        select_style_untarget(style_folder, video_path, class_info, attack_id)


if __name__ == '__main__':
    main()
