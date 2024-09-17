import os
import multiprocessing as mp
import math
import errno
from PIL import Image


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def _assemble_predictions(chunk, section_data):
    for item in chunk:
        dirpath, fname = os.path.split(item)
        img_basename = os.path.basename(fname)
      #  print(item)
       # print(img_basename)
        secdata_cur = section_data[item]

        img_assembled = Image.new('RGB', (secdata_cur['fullsize'][0],secdata_cur['fullsize'][1] ))
        for i, offset in enumerate(secdata_cur['offsets']):
            sec_fname = dirpath +'/preds/' + img_basename + '_' + str(i) + '_mask.png'
            img_sec = Image.open(sec_fname)
            img_assembled.paste(img_sec, (offset[0], offset[1]))

        dirpath_out, fname = os.path.split(secdata_cur['fullpath'])
        dirpath_out = dirpath_out + "/preds/"
        if not os.path.isdir(dirpath_out):
            try:
                os.makedirs(dirpath_out)  #race condition which can result in an error if another process made teh directory already
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
                pass

        fullpath_out = dirpath_out + img_basename + '_pred.png'
        img_assembled.save(fullpath_out)


def assemble_predictions(section_data, params):
    n_proc = params['workers']
    jobs = []

    for chunk in chunks(section_data.keys(), math.ceil(len(section_data.keys())/n_proc)):
        j = mp.Process(target=_assemble_predictions, args=(chunk, section_data))
        j.start()
        jobs.append(j)

    for j in jobs:
        j.join()