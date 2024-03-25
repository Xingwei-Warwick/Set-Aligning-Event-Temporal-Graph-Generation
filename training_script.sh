python first_finetune.py \
    --model_name_or_path google/flan-t5-base \
    --data_path data/NYT_des_train.json \
    --output_dir fine_tuned_model_dir \
    --num_train_epochs 50 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --max_target_length 1024


accelerate launch --num_processes=4 caliberation_finetune.py --train_file data/NYT_des_train.json \
    --checkpointing_steps epoch \
    --pad_to_max_length \
    --model_name_or_path fine_tuned_model_dir-final  \
    --per_device_train_batch_size 1 \
    --line_by_line True \
    --max_seq_length 2048 \
    --num_train_epochs 3 \
    --output_dir calib_model_dir

