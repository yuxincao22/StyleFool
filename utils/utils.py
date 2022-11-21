import random
import math
import os
import numpy as np
import torch.nn as nn
import torch
import json
import cv2
from utils.color import *

R = 100
angle = 30
h0 = R * math.cos(angle / 180 * math.pi)
r0 = R * math.sin(angle / 180 * math.pi)


def video_to_images(path, crop_size=112):
    video = cv2.VideoCapture(path)
    img_data = []
    cnt = 0
    while (video.isOpened()):
        if cnt >= 16:
            break
        ret, frame = video.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tmp_data = np.array(rgb_frame)

        img = tmp_data.copy()
        if img.shape[1] == crop_size and img.shape[0] == crop_size:
            pass
        else:
            if (img.shape[1] > img.shape[0]):
                scale = float(crop_size) / float(img.shape[0])
                img = np.array(cv2.resize(np.array(img), (int(img.shape[1] * scale + 1), crop_size))).astype(np.float32)
            else:
                scale = float(crop_size) / float(img.shape[1])
                img = np.array(cv2.resize(np.array(img), (crop_size, int(img.shape[0] * scale + 1)))).astype(np.float32)
        crop_x = int((img.shape[0] - crop_size) / 2)
        crop_y = int((img.shape[1] - crop_size) / 2)
        img = img[crop_x:crop_x + crop_size, crop_y:crop_y + crop_size]
        rgb_frame = np.array(img)
        img_data.append(rgb_frame)
        cnt = cnt + 1
    frames = img_data
    video.release()
    return frames

def DistanceOf(hsv1, hsv2):
    x1 = r0 * hsv1[2] * hsv1[1] * math.cos(hsv1[0] / 180 * math.pi)
    y1 = r0 * hsv1[2] * hsv1[1] * math.sin(hsv1[0] / 180 * math.pi)
    z1 = h0 * (1 - hsv1[2])
    x2 = r0 * hsv2[2] * hsv2[1] * math.cos(hsv2[0] / 180 * math.pi)
    y2 = r0 * hsv2[2] * hsv2[1] * math.sin(hsv2[0] / 180 * math.pi)
    z2 = h0 * (1 - hsv2[2])
    dx = x1 - x2
    dy = y1 - y2
    dz = z1 - z2
    return math.sqrt(dx * dx + dy * dy + dz * dz)

def RGB2HSV(rgb):
    (red, green, blue) = rgb
    r = red / 255.0
    g = green / 255.0
    b = blue / 255.0

    ma = max(r, max(g, b))
    mi = min(r, min(g, b))

    hue = 0.0
    if (ma == r and g >= b):
        if (ma - mi == 0):
            hue = 0.0
        else:
            hue = 60 * (g - b) / (ma - mi)
    elif (ma == r and g < b):
        hue = 60 * (g - b) / (ma - mi) + 360
    elif (ma == g):
        hue = 60 * (b - r) / (ma - mi) + 120
    elif (ma == b):
        hue = 60 * (r - g) / (ma - mi) + 240
    sat = 0.0 if (ma == 0) else (1.0 - mi / ma)
    bri = ma
    return np.array([hue, sat, bri])

def RGB2HSV_batch(rgbs):
    output = []
    for rgb in rgbs:
        (red, green, blue) = rgb
        r = red / 255.0
        g = green / 255.0
        b = blue / 255.0

        ma = max(r, max(g, b))
        mi = min(r, min(g, b))

        hue = 0.0
        if (ma == r and g >= b):
            if (ma - mi == 0):
                hue = 0.0
            else:
                hue = 60 * (g - b) / (ma - mi)
        elif (ma == r and g < b):
            hue = 60 * (g - b) / (ma - mi) + 360
        elif (ma == g):
            hue = 60 * (b - r) / (ma - mi) + 120
        elif (ma == b):
            hue = 60 * (r - g) / (ma - mi) + 240
        sat = 0.0 if (ma == 0) else (1.0 - mi / ma)
        bri = ma
        output.append(np.array([hue, sat, bri]))
    output = np.array(output)
    return output

def judge_edge(img):
    black_edge_flag = False
    (h, w, c) = img.shape
    cnt = 0
    for i in range(h):
        cnt += (sum(sum(img[i] <= 5)) == w * c)
    if cnt > 5:
        black_edge_flag = True
    return black_edge_flag

def calculate_color(style_folder, save_path, maxColor=7):
    styles = sorted(os.listdir(style_folder))

    styles_themes_info = dict()
    styles_top_rgb_info = dict()
    styles_top_hsv_info = dict()
    for style in styles:
        style_path = style_folder + style.split(".")[0] + ".png"
        print(style_path)
        pixDatas = [getPixData(style_path)]
        try:
            black_edge_flag = judge_edge(pixDatas[0])  # remove images with all non-semantic black areas in a row
            if black_edge_flag:
                a = 1 / 0 # call except
            themes = [testMMCQ(pixDatas, maxColor)] # color themes
            style_top_list = themes[0][0]
            style_top_hsv = RGB2HSV_batch(style_top_list).tolist()
            if style_top_hsv[0][2] < 0.138: # black will not become the first color theme
                a = 1 / 0 # call except
        except:
            themes = [[[]]]
            style_top_list = []
            style_top_hsv = []
        finally:
            styles_themes_info[style.split(".")[0]] = themes
            styles_top_rgb_info[style.split(".")[0]] = style_top_list
            styles_top_hsv_info[style.split(".")[0]] = style_top_hsv

    with open(save_path + "styles_themes_info.csv", "w") as f:
        json.dump(styles_themes_info, f)

    with open(save_path + "styles_top_rgb_info.csv", "w") as ff:
        json.dump(styles_top_rgb_info, ff)

    with open(save_path + "styles_top_hsv_info.csv", "w") as fff:
        json.dump(styles_top_hsv_info, fff)

def check_danger_label(styles_themes_info, styles_top_hsv_info, styles, class_info, num=101):
    # prevent that all videos of a label all have all non-semantic black areas in a row or black color theme,
    # especially in UCF-101 dataset.
    danger_id = []
    for label in range(num):
        styles_top_hsv = []
        themes_total = []
        for each_style in styles:
            if class_info[each_style.split("_")[1]] == label:
                themes = styles_themes_info[each_style.split(".")[0]]
                style_top_hsv = styles_top_hsv_info[each_style.split(".")[0]]
                if len(style_top_hsv) > 0:
                    styles_top_hsv.append(style_top_hsv)
                    themes_total.append(themes)
                else:
                    styles_top_hsv.append([[]])
                    themes_total.append([[[]]])
        if np.array(styles_top_hsv).any() == False:
            danger_id.append(label)
    return danger_id

def select_style_target(style_folder, video_path, class_info, attack_id, random_seed=20008, miu=100000):
    styles = sorted(os.listdir(style_folder))
    video_path_class = sorted(os.listdir(video_path))
    info = {}
    for subdir in video_path_class:
        sub_path = video_path + subdir + "/"
        sub_sub_npy = sorted(os.listdir(sub_path))
        for subsubdir in sub_sub_npy:
            sub_sub_path = sub_path + subsubdir
            info[subsubdir] = sub_sub_path

    with open("styles_main_color_csv/styles_themes_info.csv", "r") as f1:
        styles_themes_info = json.load(f1)
    with open("styles_main_color_csv/styles_top_rgb_info.csv", "r") as f2:
        styles_top_rgb_info = json.load(f2)
    with open("styles_main_color_csv/styles_top_hsv_info.csv", "r") as f3:
        styles_top_hsv_info = json.load(f3)
    with open("styles_main_color_csv/styles_superposition_info.csv", "r") as f4:
        styles_superposition_info = json.load(f4)

    slice = attack_id
    random.seed(random_seed)
    danger_id = check_danger_label(styles_themes_info, styles_top_hsv_info, styles, class_info, num=101)
    with open("batch_command_target.sh", "w") as f:
        for id in slice:
            npy_name = styles[id]
            video_top_hsv = styles_top_hsv_info[npy_name.split(".")[0]]
            # prevent that the attacked video has all non-semantic black areas in a row or black color theme,
            # especially in UCF-101 dataset,
            # which does not mean these videos cannot be attacked,
            # but the naturalness of the adversarial video will be slightly decreased.
            iid = id
            while video_top_hsv == []:
                if id > len(styles) / 2:
                    iid = iid - 1 if (iid - 1) >= 1 else 0
                else:
                    iid = iid + 1 if (iid + 1) <= len(styles) - 2 else len(styles) - 1
                npy_name = styles[iid]
                video_top_hsv = styles_top_hsv_info[npy_name.split(".")[0]]

            vid_path_avi = info[npy_name.split("_00")[0] + ".avi"]
            vid_class_name = vid_path_avi.split("/")[-2]
            vid_label = class_info[vid_class_name]
            # randomly choose target label
            target_label = random.randint(0, 100)
            while target_label == vid_label or target_label in danger_id:
                target_label = random.randint(0, 100)
            target_vids = []
            for each_style in styles:
                if class_info[each_style.split("_")[1]] == target_label:
                    target_vids.append(each_style)

            styles_top_hsv = []
            themes_total = []
            for style in target_vids:
                themes = styles_themes_info[style.split(".")[0]]
                style_top_hsv = styles_top_hsv_info[style.split(".")[0]]
                if len(style_top_hsv) > 0:
                    styles_top_hsv.append(style_top_hsv)
                    themes_total.append(themes)
                else:
                    styles_top_hsv.append([[]])
                    themes_total.append([[[]]])

            distances = []
            for i in range(len(styles_top_hsv)):
                score = styles_superposition_info[target_vids[i].split(".")[0]]
                if score == 0 or styles_top_hsv[i] == [[]]:
                    distances.append(1e10)
                    continue
                else:
                    score_distance = (1 - score) * miu
                total_distance = DistanceOf(video_top_hsv[0], styles_top_hsv[i][0]) + DistanceOf(video_top_hsv[1],
                                                                                                 styles_top_hsv[i][
                                                                                                     1]) + DistanceOf(
                    video_top_hsv[2], styles_top_hsv[i][2]) + score_distance
                distances.append(total_distance)
            distances = np.array(distances)
            distances_sorted = sorted(distances)
            id_sorted = np.argsort(distances)  # ascending order
            print(distances_sorted[0:10])

            best_style = target_vids[id_sorted[0]]
            style_path = style_folder + best_style
            print(style_path)

            command = "bash ./stylizeVideo_target.sh " + vid_path_avi + " " + style_path + "\n"
            print(command)
            f.writelines("echo 'begin'\n")
            f.writelines(command)
            f.writelines("echo 'end'\n\n")

def select_style_untarget(style_folder, video_path, class_info, attack_id):
    styles = sorted(os.listdir(style_folder))
    video_path_class = sorted(os.listdir(video_path))
    info = {}
    for subdir in video_path_class:
        sub_path = video_path + subdir + "/"
        sub_sub_npy = sorted(os.listdir(sub_path))
        for subsubdir in sub_sub_npy:
            sub_sub_path = sub_path + subsubdir
            info[subsubdir] = sub_sub_path

    with open("styles_main_color_csv/styles_themes_info.csv", "r") as f1:
        styles_themes_info = json.load(f1)
    with open("styles_main_color_csv/styles_top_rgb_info.csv", "r") as f2:
        styles_top_rgb_info = json.load(f2)
    with open("styles_main_color_csv/styles_top_hsv_info.csv", "r") as f3:
        styles_top_hsv_info = json.load(f3)
    with open("styles_main_color_csv/styles_superposition_info.csv", "r") as f4:
        styles_superposition_info = json.load(f4)
    slice = attack_id
    with open("batch_command_untarget.sh", "w") as f:
        for id in slice:
            npy_name = styles[id]
            video_top_hsv = styles_top_hsv_info[npy_name.split(".")[0]]
            # prevent that the attacked video has all non-semantic black areas in a row or black color theme,
            # especially in UCF-101 dataset,
            # which does not mean these videos cannot be attacked,
            # but the naturalness of the adversarial video will be slightly decreased.
            iid = id
            while video_top_hsv == []:
                if id > len(styles) / 2:
                    iid = iid - 1 if (iid - 1) >= 1 else 0
                else:
                    iid = iid + 1 if (iid + 1) <= len(styles) - 2 else len(styles) - 1
                npy_name = styles[iid]
                video_top_hsv = styles_top_hsv_info[npy_name.split(".")[0]]

            vid_path_avi = info[npy_name.split("_00")[0] + ".avi"]
            vid_class_name = vid_path_avi.split("/")[-2]
            vid_label = class_info[vid_class_name]

            target_vids = []
            for each_style in styles:
                if class_info[each_style.split("_")[1]] != vid_label:
                    target_vids.append(each_style)

            styles_top_hsv = []
            themes_total = []
            for style in target_vids:
                themes = styles_themes_info[style.split(".")[0]]
                style_top_hsv = styles_top_hsv_info[style.split(".")[0]]
                if len(style_top_hsv) > 0:
                    styles_top_hsv.append(style_top_hsv)
                    themes_total.append(themes)
                else:
                    styles_top_hsv.append([[]])
                    themes_total.append([[[]]])

            video_top_hsv = styles_top_hsv_info[npy_name.split(".")[0]]

            distances = []
            for i in range(len(styles_top_hsv)):
                score = styles_superposition_info[target_vids[i].split(".")[0]]
                if score == 0 or styles_top_hsv[i] == [[]]:
                    distances.append(1e10)
                    continue
                else:
                    score_distance = (1 - score) * 0
                total_distance = DistanceOf(video_top_hsv[0], styles_top_hsv[i][0]) + DistanceOf(video_top_hsv[1],
                                                                                                 styles_top_hsv[i][
                                                                                                     1]) + DistanceOf(
                    video_top_hsv[2], styles_top_hsv[i][2]) + score_distance
                distances.append(total_distance)
            distances = np.array(distances)
            distances_sorted = sorted(distances)
            id_sorted = np.argsort(distances)  # ascending order
            print(distances_sorted[0:10])

            best_style = target_vids[id_sorted[0]]
            style_path = style_folder + best_style
            print(style_path)

            command = "bash ./stylizeVideo_untarget.sh " + vid_path_avi + " " + style_path + "\n"
            print(command)
            f.writelines("echo 'begin'\n")
            f.writelines(command)
            f.writelines("echo 'end'\n\n")

def preprocess(images, crop_size=112):
    input = torch.zeros(16, 3, crop_size, crop_size)
    for i in range(16):
        np_image = images[i].data.cpu().numpy()
        np_image = np.squeeze(np.transpose(np_image, (0, 2, 3, 1)))
        np_image_new = (np_image - np.array([[[90, 98, 102]]]))
        np_image_new = np.transpose(np_image_new, (2, 0, 1))
        frame = torch.from_numpy(np_image_new)
        input[i] = frame
    input = input.permute(1, 0, 2, 3)
    return input

def calculate_superposition(model, info, style_folder, save_path, device, mod, dataset):
    crop_size = 112 if mod == 'C3D' else 224
    style_list = sorted([f for f in os.listdir(style_folder) if '.DS_' not in f and f.split('.')[-1] == 'png'])

    styles_superposition_info = dict()
    correct = 0
    count = 0
    for style in style_list:
        img = np.array(cv2.imread(style_folder + style))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if (img.shape[1] > img.shape[0]):
            scale = float(crop_size) / float(img.shape[0])
            img = np.array(cv2.resize(np.array(img), (int(img.shape[1] * scale + 1), crop_size))).astype(np.float32)
        else:
            scale = float(crop_size) / float(img.shape[1])
            img = np.array(cv2.resize(np.array(img), (crop_size, int(img.shape[0] * scale + 1)))).astype(np.float32)
        crop_x = int((img.shape[0] - crop_size) / 2)
        crop_y = int((img.shape[1] - crop_size) / 2)
        img = img[crop_x:crop_x + crop_size, crop_y:crop_y + crop_size]

        frames = np.tile(np.expand_dims(img, 0), (16, 1, 1, 1))

        content = frames.transpose(0, 3, 1, 2) / 255
        content = torch.tensor(content, dtype=torch.float, device='cuda')
        top_val, top1_label, logits = model(content[None, :])

        actual_label = info[style.split("_")[1]]
        print(style)
        f = nn.Softmax(dim=1)
        lo = f(logits.data)
        line = "top1_label: %s" % str(top1_label.item())
        print(line)
        line = "actual_label: %s" % str(actual_label)
        print(line)
        line = "lo[0][top1_label]: %s" % str(lo[0][top1_label].item())
        print(line)
        line = "lo[0][actual_label]: %s" % str(lo[0][actual_label].item())
        print(line)

        count += 1
        if top1_label.item() == actual_label:
            correct += 1
            styles_superposition_info[style.split(".")[0]] = lo[0][actual_label].item()
        else:
            styles_superposition_info[style.split(".")[0]] = 0
    print("correct: %d" % correct)
    print("total: %d" % count)

    with open(save_path + "styles_superposition_info.csv", "w") as f:
        json.dump(styles_superposition_info, f)
