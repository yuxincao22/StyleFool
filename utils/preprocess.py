import os
import shutil

def generate_styles(path, style_path, num=3):
    dirs = sorted(os.listdir(path))
    for dir in dirs:
        path_class = os.path.join(path, dir)
        dirs_pics = sorted(os.listdir(path_class))
        for dir_pic in dirs_pics:
            name = dir_pic
            print(name)
            path_pics = os.path.join(path_class, dir_pic)
            lists = sorted(os.listdir(path_pics))
            length = len(lists)
            if num == 1:
                selected_lists = list([0])
            else:
                interval = (length - 1) // (num - 1)
                selected_lists = list(range(0, length, interval))
            for j in selected_lists:
                orig_path = os.path.join(path_pics, lists[j])
                new_path = style_path + name + "_" + lists[j]
                print(new_path)
                shutil.copy(orig_path, new_path)
