class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/home/cv-bhlab/Documents/ML/Semantic_Seg/Benchmarks/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/path/to/datasets/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        elif dataset == 'marsh':
            return '/home/cv-bhlab/Documents/ML/Semantic_Seg/marsh_plants/Data_deeplab/real/'

        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError


class MarshClassInfo(object):
    def __init__(self):
        self.valid_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 255]
        self.class_names = ['Background','Spartina', 'Spart_dead', 'Sarcocornia',
                            'Batis', 'Juncus', 'Borrichia', 'Limonium', 'Other', 'Ignore']

        # self.ignore_index = 255
        # self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))
        self.class_map = {  # RGB to Class
            (255, 255, 255): 0,  # background
            (150, 255, 14): 0,  # Background_alt
            (127, 255, 140): 1,  # Spartina
            (113, 255, 221): 2,  # dead Spartina
            (99, 187, 255): 3,  # Sarcocornia
            (101, 85, 255): 4,  # Batis
            (212, 70, 255): 5,  # Juncus
            (255, 56, 169): 6,  # Borrichia
            (255, 63, 42): 7,  # Limonium
            (255, 202, 28): 8,  # Other
            (0  ,   0,  0): 255 # ignore
        }

def getClassInfoFactory(dataset):
    if dataset == 'marsh':
        return MarshClassInfo()

    else:
        print('Dataset {} not available.'.format(dataset))
        raise NotImplementedError
