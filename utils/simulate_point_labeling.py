# convert full semantic segmentation masks to randomly chosen point labels
# used to assess ability to use more easily obtained point labels for semantic segmentation model training.
import numpy as np
from PIL import Image
import re
import os

mask_fraction = 0.9998
reps = 1
base_directory = './real/train/'
out_dir = './out'
ss_re = re.compile(r"(.*)_mask.png")
img_ext = '.jpg'
#img_filename = './trial/train/full_2014_Row6_DSC_3203_8_mask.png'

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

for dirpath, dirs, files in os.walk(base_directory):
    for file in files:
        m = ss_re.search(file)
        if m:
            for i in range(reps):
                _tmp = np.array(Image.open(os.path.join(dirpath, file)), dtype=np.uint8)
                mask = np.random.random(_tmp.shape[0:2]) < mask_fraction
                n_labels = np.count_nonzero(np.logical_not(mask))
                print('file: {}, n_labels: {}'.format(file, n_labels))

                img_masked = _tmp
                img_masked[mask, :] = np.array([0, 0, 0])
                img_masked_PIL = Image.fromarray(img_masked)  # loading this way strips all weird png stuff from image
                out_file = os.path.join(out_dir, m.group(1) + "_" + str(i) + "_mask.png")
                img_masked_PIL.save(out_file)

                #save corresponding image
                rgb_img = Image.open(os.path.join(dirpath, m.group(1) + img_ext))
                rgb_img.save(os.path.join(out_dir, m.group(1) + "_" + str(i) + img_ext))


