#!/bin/bash
python train_torcs.py \
    --save-path mpc_5_noseg_nosupervise_pos \
    --continuous \
    --num-total-act 2 \
    --pred-step 5 \
    --use-pos \
    --use-angle \
    --sample-with-pos \
    --sample-with-angle \
    --num-same-step 1 \
    --no-supervise \
    --data-parallel \
    --id 24 \
    --resume
