import os
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='StyleFool_attack_prepare')
parser.add_argument('--output-npy-path', type=str, default='output_npy/', help='output_adversarial_npy_path')
parser.add_argument('--output-video-path', type=str, default='output_video/', help='output_adversarial_video_path')

def resize(frames, size=(112, 112), rgb2bgr=False):
    img_datas = []
    cnt = 0
    for i in range(16):
        tmp_data = np.array(frames[i])
        img = tmp_data.copy()
        if rgb2bgr == True:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = np.array(cv2.resize(np.array(img), size))
        rgb_frame = np.array(img)
        img_datas.append(rgb_frame)
        cnt = cnt + 1
    frames = np.array(img_datas)
    return frames

def main():
    args = parser.parse_args()
    output_path = args.output_video_path
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    crop_size = 112
    npy_paths = [args.output_npy_path + "adv_target/", args.output_npy_path + "adv_untarget/"]
    # adv
    for path in npy_paths:
        list = sorted(os.listdir(path))
        for lis in list:
            content = np.load(path + lis)
            if content.shape[1] == 3:
                content = np.transpose(content, (0, 2, 3, 1))
            content = resize(content)
            if "untarget" in path:
                video_dir = output_path + 'video_{}.avi'.format(lis.split("stylized")[0] + "adv_untarget")
            elif "target" in path:
                video_dir = output_path + 'video_{}.avi'.format(lis.split("stylized")[0] + "adv_target")
            _, h, w, c = content.shape
            img_size = (w, h)
            fps = 8
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            video_writer = cv2.VideoWriter(video_dir, fourcc, fps, img_size)
            for i in range(0, 16):
                s_np_image = content[i]
                if s_np_image.max() < 1.5:
                    s_np_image = s_np_image * 255
                s_np_image = np.asarray(s_np_image, np.uint8)
                s_np_image = cv2.cvtColor(s_np_image, cv2.COLOR_RGB2BGR)
                video_writer.write(s_np_image)
            video_writer.release()
            print('the video location is:', video_dir)
    # orig
    for path in npy_paths:
        list = sorted(os.listdir(path))
        for lis in list:
            if "untarget" in path:
                ppm_path = "v_untarget/" + lis.split("-stylized")[0] + "/"
            elif "target" in path:
                ppm_path = "v_target/" + lis.split("-stylized")[0] + "/"
            content = []
            for i in range(16):
                frame = cv2.imread(
                    ppm_path + "frame_%s.ppm" % (str(i * 4 + 1).zfill(4)))
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
                content.append(rgb_frame)
            content = np.array(content)
            if content.shape[0] < 16:
                continue
            if content.shape[1] == 3:
                content = np.transpose(content, (0, 2, 3, 1))
            content = resize(content)
            video_dir = output_path + 'video_{}.avi'.format(lis.split("stylized")[0] + "orig")
            _, h, w, c = content.shape
            img_size = (w, h)
            fps = 8
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            video_writer = cv2.VideoWriter(video_dir, fourcc, fps, img_size)
            for i in range(0, 16):
                s_np_image = content[i]
                if s_np_image.max() < 1.5:
                    s_np_image = s_np_image * 255
                s_np_image = np.asarray(s_np_image, np.uint8)
                s_np_image = cv2.cvtColor(s_np_image, cv2.COLOR_RGB2BGR)
                video_writer.write(s_np_image)
            video_writer.release()
            print('the video location is:', video_dir)

if __name__ == '__main__':
    main()
