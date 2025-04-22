python main.py --n_hid 512 --dataset epinions --n_layers 2 --s_layer 2 --lr 0.0003 --difflr 0.000001 --reg 0.0001 --batch_size 4096 --test_batch_size 1024 \
 --emb_size 4 --steps 40 --noise_scale 0.005 --model_dir './Model/epinions/' --svdlayer 1 --cuda 1 --n_epoch 150
