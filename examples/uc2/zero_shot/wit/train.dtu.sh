#!/bin/bash

TASK=20
MODEL=uc2
MODEL_CONFIG=uc2_base
TASKS_CONFIG=iglue_trainval_tasks_boxes36
PRETRAINED=checkpoints/iglue/pretrain/${MODEL}/${MODEL_CONFIG}/uc2_checkpoint_200000.bin
OUTPUT_DIR=checkpoints/iglue/zero_shot/wit/${MODEL}
LOGGING_DIR=logs/iglue/wit/${MODEL_CONFIG}

source envs/iglue/bin/activate

cd ../../../..
python train_task.py \
  --bert_model huggingface/xlm-roberta-base --config_file config/${MODEL_CONFIG}.json \
  --from_pretrained ${PRETRAINED} --loss "TripletLoss" \
  --tasks_config_file config_tasks/${TASKS_CONFIG}.yml --task $TASK --gradient_accumulation_steps 4 --num_workers 20 --num_val_workers 20 \
  --adam_epsilon 1e-6 --adam_betas 0.9 0.999 --adam_correct_bias --weight_decay 0.0001 --warmup_proportion 0.1 --clip_grad_norm 1.0 \
  --optim_train_epochs 2 --eval_steps 1000 \
  --output_dir ${OUTPUT_DIR} \
  --logdir ${LOGGING_DIR} \
#  --resume_file ${OUTPUT_DIR}/RetrievalFlickr30k_${MODEL_CONFIG}/pytorch_ckpt_latest.tar
deactivate
