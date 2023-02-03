CUDA_VISIBLE_DEVICES=0,1  OMP_NUM_THREADS=1  torchrun --nproc_per_node=2 dist_train.py --net-id pgt_l --epochs 6 --dataset gen_off_hccr --save-ckpt ckpt/boost_pretrain --workers 4  --batch-size 256  --first-decay 0.7 --last-decay 0.9 --lr 0.002 --in-features 3 --save-iter 2500 --online 0
CUDA_VISIBLE_DEVICES=0,1  OMP_NUM_THREADS=1  torchrun --nproc_per_node=2 dist_train.py --net-id pgt_l --epochs 10 --dataset off_hccr  --load-ckpt  ckpt/boost_pretrain/best.ckpt --workers 4  --batch-size 256  --first-decay 0.7 --last-decay 0.9 --lr 0.002 --in-features 3 --save-iter 2500 --online 0

