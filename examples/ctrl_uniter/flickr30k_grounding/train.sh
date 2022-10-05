#!/bin/bash

TASK=18
MODEL=ctrl_uniter
MODEL_CONFIG=ctrl_uniter_base
TASKS_CONFIG=all_trainval_tasks
PRETRAINED=checkpoints/conceptual_captions/${MODEL}/${MODEL_CONFIG}/pytorch_model_9.bin
OUTPUT_DIR=checkpoints/flickr30k_grounding/${MODEL}
LOGGING_DIR=logs/flickr30k_grounding

python train_task.py \
	--config_file config/${MODEL_CONFIG}.json --from_pretrained ${PRETRAINED} \
	--tasks_config_file config_tasks/${TASKS_CONFIG}.yml --task $TASK \
	--adam_epsilon 1e-6 --adam_betas 0.9 0.999 --adam_correct_bias --weight_decay 0.0001 --warmup_proportion 0.1 --clip_grad_norm 1.0 \
	--output_dir ${OUTPUT_DIR} \
	--logdir ${LOGGING_DIR} \
	--gradient_accumulation_steps 2 \
#	--resume_file ${OUTPUT_DIR}/RetrievalFlickr30k_${MODEL_CONFIG}/pytorch_ckpt_latest.tar