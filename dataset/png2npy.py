import os
import numpy as np
import shutil
import cv2

crop_size = 112
path = "UCF-101_npy/"
class_folder = sorted(os.listdir(path))
for cla in class_folder:
    class_path = path + cla + "/"
    folder = sorted(os.listdir(class_path))
    for cl in folder:
        vid_path = class_path + cl + "/"
        npy = []
        for i in range(16):
            frame = cv2.imread(
                vid_path + "%s.png" % str(i*4+1).zfill(5))
            if frame is None:
                continue
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tmp_data = np.array(rgb_frame)
            img = tmp_data.copy()
            if img.shape[1] == 112 and img.shape[0] == 112:
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
            npy.append(rgb_frame)
        npy = np.array(npy)
        path_to_delete = vid_path
        shutil.rmtree(path_to_delete)
        np.save(class_path + cl + ".npy", npy)
        print(class_path + cl + ".npy")
