accelerate launch --num_processes=2 evaluation.py --model-checkpoint calib_model_dir \
    --test-path data/NYT_des_test.json \
    --output_dir out \
    --generation beam

python compute_metrics.py --path out/calib_model_dir-nyt-test.json --num-split 2
