python train.py --model UNet --lr 0.007 --workers 1 --epochs 100 --batch-size 2 --gpu-ids 0 --checkname UNET_init --eval-interval 1 --dataset marsh --use-balanced-weights

#python train.py --backbone resnet --lr 0.007 --workers 1 --epochs 1 --batch-size 4 --gpu-ids 0 --checkname deeplab-resnet --eval-interval 1 --dataset marsh
