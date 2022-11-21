import torch
from models import C3D
from model_wrapper.vid_model_top_k import I3D_K_Model, C3D_K_Model
from pytorch_i3d import InceptionI3d


def model_initial(model, dataset, device):
    if model == 'C3D' and dataset == 'UCF101':
        model = C3D(num_classes=101, pretrained=False).cuda().to(device)
        checkpoint = torch.load(
            'ckpt/C3D-ucf101_epoch-49.pth.tar',
            map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        model = C3D_K_Model(model)
    elif model == 'C3D' and dataset == 'HMDB51':
        model = C3D(num_classes=51, pretrained=False).cuda().to(device)
        checkpoint = torch.load(
            'ckpt/C3D-hmdb51_epoch-49.pth.tar',
            map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        model = C3D_K_Model(model)
    elif model == 'I3D' and dataset == 'UCF101':
        i3d = InceptionI3d(101, in_channels=3)
        i3d.load_state_dict(torch.load('ckpt/UCF101_I3D_000500.pt'))
        i3d.cuda().to(device)
        i3d.train(False)
        i3d.eval()
        model = I3D_K_Model(i3d)
    elif model == 'I3D' and dataset == "HMDB51":
        i3d = InceptionI3d(51, in_channels=3)
        i3d.load_state_dict(torch.load('ckpt/HMDB51_I3D_000500.pt'))
        i3d.cuda().to(device)
        i3d.train(False)
        i3d.eval()
        model = I3D_K_Model(i3d)
    return model
