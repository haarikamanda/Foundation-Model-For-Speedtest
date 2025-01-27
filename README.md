This module takes the preprocessed dataset using the pre_process module step as input and uses different standardized library to train the learning models (pretraining and finetuning). 
* The sample script for pretraining is
`python3 NetfoundPretraining.py --train_file /data/UCSBwithMetaPretrainingNoOPTProto.csv --output_dir /data/Netfound-pretrain-mlm-port-1024-meta-proto-NoOpt-flat/ --eval_steps 10000 --learning_rate 5e-6 --do_eval --evaluation_strategy steps --save_steps 10000 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --overwrite_output_dir  --mlm_probability 0.15 --max_eval_samples 100000 --do_train`

* The following is the sample script for finetuning the pretrained model.
`python3 NetfoundFinetuning.py --train_file /data/UCSBwithMetaFinetuningNoOPT_train.csv --test_file /data/UCSBwithMetaFinetuningNoOPT_test.csv --output_dir <output_dir> --learning_rate 2e-5 --do_eval --evaluation_strategy epoch --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --model_name_or_path <pretrained_model> --overwrite_output_dir --num_labels <num_labels_in_dataset> --load_best_model_at_end --num_train_epochs 5 --save_strategy epoch --do_train`
