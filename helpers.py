import os
import torch
import torchvision
from PIL import Image
import dataloaders

def write_predictions_rgb(pred, img_fnames, dim, dataset):
    pred_rgb = dataloaders.utils.encode_seg_map_sequence(pred, dataset,
                                                             dtype='int')  # convert predictions to rgb masks

    dim = torch.cat((dim[0].unsqueeze(1), dim[1].unsqueeze(1)), dim=1)
    # write out masks (resize to original image dimensions)
    for fname, d, mask in zip(img_fnames, dim, pred_rgb):
        w = d[0]
        h = d[1]
        mask = torchvision.transforms.ToPILImage()(mask)
        mask = mask.resize((w, h), Image.NEAREST)

        base_path, fn = os.path.split(fname)
        base_fn, ext = os.path.splitext(fn)
        outdir = base_path + "/preds/"
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        outpath = outdir + base_fn + "_preds.png"
        mask.save(outpath)
