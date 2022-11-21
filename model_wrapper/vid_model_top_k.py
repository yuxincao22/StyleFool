import torch
import torch.nn as nn
import numpy as np

class InceptionI3D_K_Model():
    def __init__(self, model):
        self.k = 1
        self.model = model

    def set_k(self, k):
        self.k = k

    def preprocess(self, vid):
        vid_t = vid.clone()
        vid_t.mul_(2).sub_(1)
        vid_t = vid_t.permute(0, 2, 1, 3, 4)
        return vid_t

    def get_top_k(self, vid, k):
        with torch.no_grad():
            out = self.model(self.preprocess(vid))
        logits = out.mean(2)
        top_val, top_idx = torch.topk(nn.functional.softmax(logits, 1), k)
        return top_val, top_idx, logits

    def __call__(self, vid):
        return self.get_top_k(vid, self.k)


class C3D_K_Model():
    def __init__(self, model):
        self.k = 1
        self.model = model

    def set_k(self, k):
        self.k = k

    def preprocess(self, vid):
        # input: batch * 16 * 3 * 112 * 112
        images = vid.clone()
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        inputs = torch.zeros((images.shape[0], 3, 16, 112, 112))
        for ii in range(images.shape[0]):
            imgs = images[ii].clone().data.cpu().numpy()
            input = torch.zeros(16, 3, 112, 112)
            for i in range(16):
                np_image = imgs[i]
                np_image = np.clip(np.transpose(np_image, (1, 2, 0)) * 255, 0, 255)
                np_image_new = (np_image - np.array([[[90, 98, 102]]]))
                np_image_new = np.transpose(np_image_new, (2, 0, 1))
                frame = torch.from_numpy(np_image_new)
                input[i] = frame
            input = input.permute(1, 0, 2, 3)
            inputs[ii] = input
        vid_t = inputs.cuda()
        return vid_t

    def get_top_k(self, vid, k):
        with torch.no_grad():
            outputs = self.model(self.preprocess(vid))
        # scores = outputs.data[0]
        logits = outputs
        top_val, top_idx = torch.topk(nn.functional.softmax(logits, 1), k)

        return top_val, top_idx, logits

    def __call__(self, vid):
        return self.get_top_k(vid, self.k)


class I3D_K_Model():
    def __init__(self, model):
        self.k = 1
        self.model = model

    def set_k(self, k):
        self.k = k

    def preprocess(self, vid):
        # make sure input: batch * 16 * 3 * 112 * 112
        images = vid.clone()
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if images.shape[1] == 3 and images.shape[2] == 16: # torch.Size([1, 3, 16, 224, 224])
            pass
        if images.shape[1] == 16 and images.shape[2] == 3: # torch.Size([1, 16, 3, 224, 224])
            images = images.permute(0, 2, 1, 3, 4)
        if images.detach().cpu().numpy().max() > 10:
            images = images / 255
        images = torch.clamp(images, 0.0, 1.0)
        vid_t = images.cuda()
        return vid_t

    def get_top_k(self, vid, k):
        with torch.no_grad():
            outputs = self.model(self.preprocess(vid))
        # scores = outputs.data[0]
        logits = outputs.squeeze(2)
        top_val, top_idx = torch.topk(nn.functional.softmax(logits, 1), k)

        return top_val, top_idx, logits

    def __call__(self, vid):
        return self.get_top_k(vid, self.k)