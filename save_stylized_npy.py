import argparse
import cv2
import numpy as np



parser = argparse.ArgumentParser()
parser.add_argument('--styled-video-name', type=str)
parser.add_argument('--orig-video-name', type=str)
parser.add_argument('--style-img-name', type=str)
args = parser.parse_args()

styled_video_name = args.styled_video_name
styled_npy_name = "styled_npy/" + styled_video_name.split(".")[0] + ".npy"
filename = args.orig_video_name
style_filename = args.style_img_name
crop_size = 112

### stylized
fr = []
for i in range(16):
    frame = cv2.imread(
         filename + "/out-%s-%s.png" % (style_filename, str(i*4 + 1).zfill(4)))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tmp_data = np.array(rgb_frame)
    img = tmp_data.copy()
    if img.shape[1] == crop_size and img.shape[0] == crop_size:
        pass
    else:
        if (img.shape[1] > img.shape[0]):
            scale = float(crop_size) / float(img.shape[0])
            img = np.array(cv2.resize(np.array(img), (int(img.shape[1] * scale + 1), crop_size))).astype(
                np.float32)
        else:
            scale = float(crop_size) / float(img.shape[1])
            img = np.array(cv2.resize(np.array(img), (crop_size, int(img.shape[0] * scale + 1)))).astype(
                np.float32)
    crop_x = int((img.shape[0] - crop_size) / 2)
    crop_y = int((img.shape[1] - crop_size) / 2)
    img = img[crop_x:crop_x + crop_size, crop_y:crop_y + crop_size, :]
    rgb_frame = np.array(img)
    fr.append(rgb_frame)

fr = np.array(fr)
np.save(styled_npy_name, fr)
