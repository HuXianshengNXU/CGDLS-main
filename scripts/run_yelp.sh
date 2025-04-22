python main.py --n_hid 512 --dataset yelp --n_layers 2 --s_layer 2 --lr 0.007 --difflr 0.00001 --reg 0.005 --batch_size 4096 --test_batch_size 2048 \
 --emb_size 512 --steps 50 --noise_scale 0.1 --model_dir './Model/yelp/' --svdlayer 1 --cuda 0 --n_epoch 300
