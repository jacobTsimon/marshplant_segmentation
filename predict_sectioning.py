import argparse
import os
import re
import numpy as np
from tqdm import tqdm
import yaml
import shutil
import cv2
import torchvision

from PIL import Image
from dataloaders import make_data_loader
from modeling.deeplab import *
import dataloaders.utils
from utils.image_sectioning import section_images
from utils.assemble_predictions import assemble_predictions

path_regex = re.compile('.+?/(.*)$')

n_classes =9
class_labels = ['Background', 'Spartina', 'Spartina_dead', 'Sarcocornia', 'Batis', 'Juncus', 'Borrichia',
                'Limonium', 'Other']
class_map = {  # RGB to Class
    (0, 0, 0): -1,  # out of bounds
    (255, 255, 255): 0,  # background
    (150, 255, 14): 0,  # Background_alt
    (127, 255, 140): 1,  # Spartina
    (113, 255, 221): 2,  # dead Spartina
    (99, 187, 255): 3,  # Sarcocornia
    (101, 85, 255): 4,  # Batis
    (212, 70, 255): 5,  # Juncus
    (255, 56, 169): 6,  # Borrichia
    (255, 63, 42): 7,  # Limonium
    (255, 202, 28): 8  # Other
}

def maskrgb_to_class(mask, class_map):
    """ decode rgb mask to classes using class map"""
    h, w, channels = mask.shape[0], mask.shape[1], mask.shape[2]
    mask_out = -1 * np.ones((h, w), dtype=int)

    for k in class_map:
        matches = np.zeros((h, w, channels), dtype=bool)

        for c in range(channels):
            matches[:, :, c] = mask[:, :, c] == k[c]

        matches_total = np.sum(matches, axis=2)
        valid_idx = matches_total == channels
        mask_out[valid_idx] = class_map[k]

    return mask_out


def setup_argparser():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Prediction")
    parser.add_argument('--model', type=str, default=None, help='provide path to model')
    parser.add_argument('--dataset', type=str, default='marsh',
                        choices=['pascal', 'coco', 'cityscapes','marsh'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--dataset_path', type=str, default=None, help='provide path to dataset')
    parser.add_argument('--img_dim', type=int, nargs='+', default=None, help='image dimensions as width height')
    parser.add_argument('--img_sections', type=int, nargs='+', default=None, help='number of sections to cut image into as columns rows')
    parser.add_argument('--batch-size', type=int, default=2,
                        metavar='N', help='input batch size for prediction (default: 2)')
    parser.add_argument('--workers', type=int, default=8,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    return parser

def parse_model_configfile(args):
    model_data_path = os.path.join(args.model, 'model_data.yaml')
    model_data = yaml.safe_load( open(model_data_path,'r') )
    args.backbone = model_data['backbone']
    args.out_stride = model_data['out_stride']
    args.model_path = os.path.join(args.model, model_data['model_path'])
    #args.crop_size = model_data['crop_size']
    return args

def setup_img_sectioning_params(args):
    re_fbase = re.compile('^(.*)\.[jJ][pP][eE]?[gG]')

    img_dim = args.img_dim
    section_dim = args.img_sections
    patch_dim = [int(img_dim[0]/section_dim[0]), int(img_dim[1]/section_dim[1])]

    args.crop_size = patch_dim

    pred_format = "{}\t{:4.3f}\t{:5.1f}\t{:5.1f}\t{:5.1f}\t{:5.1f}\n"
    params = {'section_dim': section_dim, 'crop_size': patch_dim,  'fmt': pred_format, 're_fbase': re_fbase, 'workers': args.workers, 'write_imgs': True}

    #setup temporary folder to store section data
    tmp_folder = './tmp'  # start with a clean tmp folder
    if os.path.isdir(tmp_folder):
        shutil.rmtree(tmp_folder)
        os.mkdir(tmp_folder)
    else:
        os.mkdir(tmp_folder)
    params['outfld'] = tmp_folder

    return params, args

def make_predictions(model, dataloader, args):
    model.eval()
    if args.cuda:
        model.cuda()

    # process samples
    tbar = tqdm(dataloader, desc='predictions')
    with torch.no_grad():
        for i, sample in enumerate(tbar):

            # unpackage sample
            image = sample['image']
            label = sample['label']
            dim = sample['dim']
            dim = torch.cat((dim[0].unsqueeze(1), dim[1].unsqueeze(1)), dim=1)

            if args.cuda:
                image = image.cuda()

            # forward pass through model and make predictions
            output = model(image)
            pred = output.data.cpu().numpy()
            pred = np.argmax(pred, axis=1)

            pred_rgb = dataloaders.utils.encode_seg_map_sequence(pred, args.dataset,
                                                                 dtype='int')  # convert predictions to rgb masks

            # write out masks (resize to original image dimensions)
            for lbl, d, mask in zip(label, dim, pred_rgb):
                w = d[0]
                h = d[1]
                mask = torchvision.transforms.ToPILImage()(mask)
                mask = mask.resize((w, h), Image.NEAREST)

                base_path, fn = os.path.split(lbl)
                base_fn, ext = os.path.splitext(fn)
                outdir = base_path + "/preds/"
                if not os.path.isdir(outdir):
                    os.mkdir(outdir)
                outpath = outdir + base_fn + "_mask.png"
                mask.save(outpath)

def class_statistics(section_data, outfile):

    class_stats = {}
    for img_id in section_data: #this is slow - parallelize
        dirpath, fname = os.path.split(section_data[img_id]['fullpath'])
        name_ext = os.path.splitext(fname)
        dirpath = dirpath + "/preds/"
        fullpath = dirpath + name_ext[0] + '_pred.png'

        img_pred = cv2.imread(fullpath)
        img_pred = cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)
        img_class = maskrgb_to_class(img_pred, class_map)

        pixel_count = []
        for i in range(n_classes):
            t = np.sum(img_class == i)
            pixel_count.append(t)

        class_stats[fullpath] = pixel_count

    fout = open(outfile, 'w')
    header = 'Image_ID\t' + '\t'.join([str(i) for i in class_labels]) + '\n'
    fout.write(header)
    for img_id in class_stats:
        fout.write('{}'.format(img_id))
        for elm in class_stats[img_id]:
            fout.write('\t{:d}'.format(elm))
        fout.write('\n')

    fout.close()


def main():
    parser = setup_argparser()
    args = parser.parse_args()
    args.train = False
    args = parse_model_configfile(args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    #setup dataset
    params, args = setup_img_sectioning_params(args)
    print('sectioning images for prediction')
    section_data = section_images(args.dataset_path, params)
    #print(section_data)

    kwargs = {'num_workers': args.workers, 'pin_memory': True}
    m = path_regex.findall(args.dataset_path)
    dirpath_sub = m[0]
    root_dir = os.path.join(params['outfld'], dirpath_sub)
    print('dirpath_sub {}'.format(dirpath_sub ))
    dataloader, nclass = make_data_loader(args, root=root_dir, **kwargs)

    # setup model
    model = DeepLab(num_classes=nclass,
                    backbone=args.backbone,
                    output_stride=args.out_stride)

    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['state_dict'])

    #make predictions for dataset using model
    make_predictions(model, dataloader, args)

    print('assembling predicted images')
    assemble_predictions(section_data, params)

    class_statistics(section_data, 'class_cover.txt')

if __name__ == "__main__":
   main()