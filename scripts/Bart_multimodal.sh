python ./src/run.py \
        -model=multi_modal_bart \
        -train_src_path=./dataset/train_tran.tok.txt \
        -train_tgt_path=./dataset/train_desc.tok.txt \
        -val_src_path=./dataset/val_tran.tok.txt \
        -val_tgt_path=./dataset/val_desc.tok.txt \
        -test_src_path=./dataset/test_tran.tok.txt \
        -test_tgt_path=./dataset/test_desc.tok.txt \
        -image_feature_path=./dataset/video_features/ \
        -val_save_file=./evaluation/temp_valid_file \
        -test_save_file=./evaluation/results/summaries.txt \
        -log_name=multi_modal_bart \
        -gpus='0' \
        -batch_size=4 \
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
        -do_train=True \
        -do_test=False \
        -limit_val_batches=1 \
        -val_check_interval=1 \
        -img_lr_factor=5 \
        -checkpoint=None \
        -use_forget_gate \
        -cross_attn_type=0 \
        -use_img_trans \
        # -fusion_in_decoding


find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf