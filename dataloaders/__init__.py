from dataloaders.datasets import cityscapes, coco, combine_dbs, pascal, sbd, saltmarsh
from torch.utils.data import DataLoader

def make_data_loader(args, root = './', **kwargs):

    if args.dataset == 'pascal':
        train_set = pascal.VOCSegmentation(args, split='train')
        val_set = pascal.VOCSegmentation(args, split='val')
        if args.use_sbd:
            sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
            train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'cityscapes':
        train_set = cityscapes.CityscapesSegmentation(args, split='train')
        val_set = cityscapes.CityscapesSegmentation(args, split='val')
        test_set = cityscapes.CityscapesSegmentation(args, split='test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'coco':
        train_set = coco.COCOSegmentation(args, split='train')
        val_set = coco.COCOSegmentation(args, split='val')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class

    elif args.dataset=='marsh':
        if args.train:
            train_set = saltmarsh.SaltmarshSegmentation(args, split='train')
            val_set = saltmarsh.SaltmarshSegmentation(args, split='val')
            test_set = saltmarsh.SaltmarshSegmentation(args, split='test')

            num_class = train_set.NUM_CLASSES
            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs) #made drop last true because of non-divisible number with batch size.
            val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False, **kwargs)
            test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, **kwargs)
            return train_loader, val_loader, test_loader, num_class
        else:
            predict_set = saltmarsh.SaltmarshSegmentation(args, root=root, split='predict')
            predict_loader = DataLoader(predict_set, batch_size=args.batch_size, shuffle=False, drop_last=False, **kwargs)
            return predict_loader, predict_set.NUM_CLASSES

    else:
        raise NotImplementedError
