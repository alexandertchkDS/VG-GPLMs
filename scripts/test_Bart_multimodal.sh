NCCL_DEBUG=WARN NCCL_NET_GDR_LEVEL=0 python ./src/run.py \
        -model=multi_modal_bart \
        -train_src_path=./dataset/sum_all/train_tran.tok.txt \
        -train_tgt_path=./dataset/sum_all/train_desc.tok.txt \
        -val_src_path=./dataset/sum_all/val_tran.tok.txt \
        -val_tgt_path=./dataset/sum_all/val_desc.tok.txt \
        -test_src_path=./dataset/sum_all/test_tran.tok.txt \
        -test_tgt_path=./dataset/sum_all/test_desc.tok.txt \
        -image_feature_path=./dataset/video_action_features/ \
        -val_save_file=./evaluation/temp_valid_file \
        -test_save_file=./evaluation/results/summaries.txt \
        -log_name=test \
        -gpus='0' \
        -batch_size=3 \
        -learning_rate=3e-5 \
        -scheduler_lambda1=10 \
        -scheduler_lambda2=0.95 \
        -num_epochs=100 \
        -grad_accumulate=5 \
        -max_input_len=512 \
        -max_output_len=64 \
        -max_img_len=256 \
        -n_beams=5 \
        -random_seed=0 \
        -do_train=False \
        -do_test=True \
        -limit_val_batches=1 \
        -val_check_interval=1 \
        -img_lr_factor=5 \
        -checkpoint=./lightning_logs/multi_modal_bart/default/version_13/epoch=98-step=4652.ckpt \
        -use_forget_gate \
        -cross_attn_type=0 \
        -use_img_trans \


find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf