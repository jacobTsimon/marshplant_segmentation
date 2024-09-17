from torch.utils import data
import torch
import os
import numpy as np
import scipy.misc as m
from PIL import Image
from myinfo import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr

import sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
import myinfo

class SaltmarshSegmentation(data.Dataset):
    NUM_CLASSES = 9

    def __init__(self, args,root=Path.db_root_dir('marsh'), split="train"):

        self.root = root
        self.split = split
        self.args = args
        self.images = []
        self.masks=[]

        self.set_data_names()

        class_info = myinfo.getClassInfoFactory('marsh')
        self.valid_classes = class_info.valid_classes
        self.class_names = class_info.class_names
        self.class_map = class_info.class_map

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        
        #img_path = self.root+"/"+self.split+"/"+self.images[index]
        _img = Image.open(self.images[index]).convert('RGB')
        img_dim = _img.size

        _target = []
        if not self.split == 'predict':
           # lbl_path = self.root+"/"+self.split+"/"+self.masks[index]
            _tmp = np.array(Image.open(self.masks[index]), dtype=np.uint8)
            _target = Image.fromarray(np.array(_tmp))
        else:
            _target = self.images[index]

        sample = {'image': _img, 'label': _target }

        if self.split == 'train':
            sample =  self.transform_tr(sample)
        elif self.split == 'val':
            sample = self.transform_val(sample)
        elif self.split == 'test':
            sample =  self.transform_ts(sample)
        elif self.split == 'predict':
            sample['image'] = self.transform_pred(sample['image'])

        if not self.split == 'predict':
            sample['label'] = self.maskrgb_to_class(
                                    np.array(sample['label'], dtype=np.uint8)
                                    )
        sample['img_fname'] = self.images[index]
        sample['dim'] = img_dim

        return sample

    def set_data_names(self):
        i = 0
        #search_dir = self.root+"/"+self.split+"/"
        #print("searching dir: {}".format(self.root))
      #  print("root dir: {}".format(self.root))
        base_dir = self.root+self.split+"/"
        if self.split == 'predict':
            base_dir = self.root

       # print("searching dir: {}".format(base_dir))
        for file in os.listdir(base_dir):

            if (file.endswith("mask.png")):
                    self.masks.append( os.path.join(base_dir, file) )
                    s=file.split('_')
                    imgname="_".join(s[:-1])+".jpg"
                  #  print(imgname)
                    self.images.append( os.path.join(base_dir, imgname) )
                    i  = i +1
            elif self.split == 'predict':
                if(file.endswith(".jpg")):
                    self.images.append( os.path.join(base_dir, file) )
                    i = i + 1
        print('total images loaded: {}'.format(i))
        return True

    def maskrgb_to_class(self, mask):
     #   print('----mask->rgb----')
        mask = torch.from_numpy(np.array(mask))
        mask = torch.squeeze(mask)  # remove 1

        # check the present values in the mask, 0 and 255 in my case
        #print('unique values rgb    ', torch.unique(mask))
        # -> unique values rgb     tensor([  0, 255], dtype=torch.uint8)

        class_mask = mask
        class_mask = class_mask.permute(2, 0, 1).contiguous()  #channels dim 0
        h, w = class_mask.shape[1], class_mask.shape[2]
        mask_out = torch.zeros(h, w, dtype=torch.long)

        for k in self.class_map:
            idx = (class_mask == torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
            validx = (idx.sum(0) == 3)  #must agree in all 3 channels
            mask_out[validx] = torch.tensor(self.class_map[k], dtype=torch.long)

        # check the present values after mapping, in my case 0, 1, 2, 3
      #  print('unique values mapped ', torch.unique(mask_out))
        # -> unique values mapped  tensor([0, 1, 2, 3])

        return mask_out

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size, fill=255),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_ts(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixedResize( self.args.crop_size ),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_pred(self, image):
        resize = []
        if len(self.args.crop_size) == 1:
            resize = transforms.Resize(size=(self.args.crop_size, self.args.crop_size))
        elif len(self.args.crop_size) == 2:
            resize = transforms.Resize(size=(self.args.crop_size[1], self.args.crop_size[0]))  #resize is height, width but crop_size data is width, height
        else:
            raise "crop_size should be len 1 or 2"

        composed_transforms = transforms.Compose([
            resize,
            #transforms.Resize(size=(800, 1200)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ]
        )

        return composed_transforms(image)