#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python train.py --ver gru --task I50 --hid 256 --epoch 5 --lr 0.001 --time 0 --cuda &
CUDA_VISIBLE_DEVICES=1 python train.py --ver gru --task I50 --hid 256 --epoch 5 --lr 0.0003 --time 0 --cuda &
CUDA_VISIBLE_DEVICES=1 python train.py --ver gru --task I50 --hid 256 --epoch 5 --lr 0.0001 --time 0 --cuda &

CUDA_VISIBLE_DEVICES=1 python train.py --ver retain --task I50 --hid 256 --epoch 5 --lr 0.001 --time 0 --cuda &
CUDA_VISIBLE_DEVICES=1 python train.py --ver retain --task I50 --hid 256 --epoch 5 --lr 0.0003 --time 0 --cuda &
CUDA_VISIBLE_DEVICES=1 python train.py --ver retain --task I50 --hid 256 --epoch 5 --lr 0.0001 --time 0 --cuda &
