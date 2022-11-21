import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from utils.MMCQ import MMCQ

def imgPalette(imgs, themes, titles):
    N = len(imgs)
    fig = plt.figure()
    gs = gridspec.GridSpec(len(imgs), len(themes)+1)
    for i in range(N):
        im = fig.add_subplot(gs[i, 0])
        im.imshow(imgs[i])
        im.set_title("Image %s" % str(i+1))
        im.xaxis.set_ticks([])
        im.yaxis.set_ticks([])
        t = 1
        for themeLst in themes:
            theme = themeLst[i]
            pale = np.zeros(imgs[i].shape, dtype=np.uint8)
            h, w, _ = pale.shape
            ph = h / len(theme)
            for y in range(h):
                pale[y, :, :] = np.array(theme[int(y / ph)], dtype=np.uint8)
            pl = fig.add_subplot(gs[i, t])
            pl.imshow(pale)
            pl.set_title(titles[t-1])
            pl.xaxis.set_ticks([])
            pl.yaxis.set_ticks([])
            t += 1
    plt.show()

def getPixData(imgfile):
    return cv2.cvtColor(cv2.imread(imgfile, 1), cv2.COLOR_BGR2RGB)

def testMMCQ(pixDatas, maxColor):
    themes = list(map(lambda d: MMCQ(d, maxColor).quantize(), pixDatas))
    return themes
