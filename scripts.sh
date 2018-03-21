#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python train.py --ver gru --task H26 --hid 128 --epoch 5 --lr 0.001 --time 0 --cuda &
CUDA_VISIBLE_DEVICES=0 python train.py --ver retain --task H26 --hid 128 --epoch 5 --lr 0.001 --time 0 --cuda &
CUDA_VISIBLE_DEVICES=0 python train.py --ver ex --task H26 --hid 128 --epoch 5 --lr 0.001 --time 0 --cuda &
CUDA_VISIBLE_DEVICES=0 python train.py --ver ex --task H26 --hid 128 --epoch 5 --lr 0.001 --time 1 --cuda &

CUDA_VISIBLE_DEVICES=0 python train.py --ver gru --task I50 --hid 128 --epoch 5 --lr 0.001 --time 0 --cuda &
CUDA_VISIBLE_DEVICES=0 python train.py --ver retain --task I50 --hid 128 --epoch 5 --lr 0.001 --time 0 --cuda &
CUDA_VISIBLE_DEVICES=0 python train.py --ver ex --task I50 --hid 128 --epoch 5 --lr 0.001 --time 0 --cuda &
CUDA_VISIBLE_DEVICES=0 python train.py --ver ex --task I50 --hid 128 --epoch 5 --lr 0.001 --time 1 --cuda &

CUDA_VISIBLE_DEVICES=0 python train.py --ver ex --task H26 --hid 64 --epoch 5 --lr 0.001 --time 1 --cuda &
CUDA_VISIBLE_DEVICES=0 python train.py --ver ex --task H26 --hid 256 --epoch 5 --lr 0.001 --time 1 --cuda &
CUDA_VISIBLE_DEVICES=0 python train.py --ver ex --task I50 --hid 64 --epoch 5 --lr 0.001 --time 1 --cuda &
CUDA_VISIBLE_DEVICES=0 python train.py --ver ex --task I50 --hid 256 --epoch 5 --lr 0.001 --time 1 --cuda &