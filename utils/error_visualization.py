import numpy as np
import os
from PIL import Image
import torch

class Error_Visualizer():
    def __init__(self):
        self.dummy = []


    def by_class(self, target, pred, img_fnames, dim, target_classes):
        batch_size = target.shape[0]
        dim = torch.cat((dim[0].unsqueeze(1), dim[1].unsqueeze(1)), dim=1)
        for i in range(batch_size):
            base_path, fn = os.path.split(img_fnames[i])
            base_fn, ext = os.path.splitext(fn)
            outdir = base_path + "/error_vis/"
            if not os.path.isdir(outdir):
                os.mkdir(outdir)

            for cls in target_classes:
                trgt = target[i, : ,:] == cls
                prd  = pred[i, : ,:] == cls
                tp = np.logical_and(trgt, prd)
                fp = np.logical_and(~trgt, prd)
                fn = np.logical_and(trgt, ~prd)
                tn = np.logical_and(~trgt, ~prd) #probably not of interest but calculate for completeness

                rgb = np.zeros((target.shape[1], target.shape[2], 3), dtype=np.uint8)
                rgb[tp,:] = np.array([255, 255, 255], ndmin=3)
                rgb[fp, :] = np.array([255, 0, 0], ndmin=3)
                rgb[fn, :] = np.array([0, 0, 255], ndmin=3)

                img = Image.fromarray(rgb)
                w = dim[i][0]
                h = dim[i][1]
                img = img.resize((w, h), Image.NEAREST)

                outpath = outdir + base_fn + "_cls" + str(cls) + ".png"
                img.save(outpath)

  #  def globally(self, target, pred):

