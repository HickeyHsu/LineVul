# defect repair
python vulrepair_main.py   --model_name=t5_repair_model.bin   --output_dir=./saved_models   --tokenizer_name=/home/hickey/pretrained/Salesforce/codet5-base  --model_name_or_path=/home/hickey/pretrained/Salesforce/codet5-base  --train_data_file=../data/big-vul_dataset/train.csv   --eval_data_file=../data/big-vul_dataset/val.csv   --test_data_file=../data/big-vul_dataset/test.csv --do_train --do_test --epochs 75 --encoder_block_size 512  --decoder_block_size 256  --train_batch_size 8  --eval_batch_size 8  --learning_rate 2e-5  --max_grad_norm 1.0 --evaluate_during_training  --seed 123456  2>&1 | tee train.log
# linevul_CoText
## RQ1 Training + Inference inference
### CoText
python linevul_cotext.py   --model_name=CoText_linevul_model.bin  --output_dir=./saved_models   --model_type=t5   --tokenizer_name=/home/hickey/pretrained/razent/cotext-1-ccg   --model_name_or_path=/home/hickey/pretrained/razent/cotext-1-ccg   --do_train   --do_test   --train_data_file=../data/big-vul_dataset/train.csv   --eval_data_file=../data/big-vul_dataset/val.csv   --test_data_file=../data/big-vul_dataset/test.csv   --epochs 10   --block_size 512   --train_batch_size 16   --eval_batch_size 16   --learning_rate 2e-5   --max_grad_norm 1.0   --evaluate_during_training   --seed 123456  2>&1 | tee train.log
### Codebert
python linevul_main.py   --model_name=Codebert_linevul_model.bin  --output_dir=./saved_models   --model_type=roberta   --tokenizer_name=/home/hickey/pretrained/microsoft/codebert-base   --model_name_or_path=/home/hickey/pretrained/microsoft/codebert-base   --do_train   --do_test   --train_data_file=../data/big-vul_dataset/train.csv   --eval_data_file=../data/big-vul_dataset/val.csv   --test_data_file=../data/big-vul_dataset/test.csv   --epochs 10   --block_size 512   --train_batch_size 16   --eval_batch_size 16   --learning_rate 2e-5   --max_grad_norm 1.0   --evaluate_during_training   --seed 123456  2>&1 | tee train.log
### CodeT5
python linevul_cotext.py   --model_name=CoT5_linevul_model.bin  --output_dir=./saved_models   --model_type=t5   --tokenizer_name=/home/hickey/pretrained/Salesforce/codet5-base   --model_name_or_path=/home/hickey/pretrained/Salesforce/codet5-base   --do_train   --do_test   --train_data_file=../data/big-vul_dataset/train.csv   --eval_data_file=../data/big-vul_dataset/val.csv   --test_data_file=../data/big-vul_dataset/test.csv   --epochs 10   --block_size 512   --train_batch_size 16   --eval_batch_size 16   --learning_rate 2e-5   --max_grad_norm 1.0   --evaluate_during_training   --seed 123456  2>&1 | tee train.log
# resources download
## raw function-level predictions
cd linevul
cd results
gdown https://drive.google.com/uc?id=1WqvMoALIbL3V1KNQpGvvTIuc3TL5v5Q8

## dataset
cd data
cd big-vul_dataset

train.csv: https://drive.google.com/uc?id=1ldXyFvHG41VMrm260cK_JEPYqeb6e6Yw
val.csv: https://drive.google.com/uc?id=1yggncqivMcP0tzbh8-8Eu02Edwcs44WZ
test.csv: https://drive.google.com/uc?id=1h0iFJbc5DGXCXXvvR6dru_Dms_b2zW4V
whole (i.e., train+val+test) unsplit dataset dataset: https://drive.google.com/uc?id=10-kjbsA806Zdk54Ax8J3WvLKGTzN8CMX

# replication
## RQ1
### RQ1 inference
python linevul_main.py   --model_name=12heads_linevul_model.bin   --output_dir=./saved_models   --model_type=roberta   --tokenizer_name=microsoft/codebert-base   --model_name_or_path=microsoft/codebert-base   --do_test   --train_data_file=../data/big-vul_dataset/train.csv   --eval_data_file=../data/big-vul_dataset/val.csv   --test_data_file=../data/big-vul_dataset/test.csv   --block_size 512   --eval_batch_size 512

### RQ1 Training + Inference inference
cd linevul
python linevul_main.py   --output_dir=./saved_models   --model_type=roberta   --tokenizer_name=microsoft/codebert-base   --model_name_or_path=microsoft/codebert-base   --do_train   --do_test   --train_data_file=../data/big-vul_dataset/train.csv   --eval_data_file=../data/big-vul_dataset/val.csv   --test_data_file=../data/big-vul_dataset/test.csv   --epochs 10   --block_size 512   --train_batch_size 16   --eval_batch_size 16   --learning_rate 2e-5   --max_grad_norm 1.0   --evaluate_during_training   --seed 123456  2>&1 | tee train.log

### RQ1 BoW+RF
cd bow_rf
mkdir saved_models
python rf_main.py

## RQ2
### RQ2 Top-10 Accuracy and IFA

cd linevul
python linevul_main.py   --model_name=12heads_linevul_model.bin   --output_dir=./saved_models   --model_type=roberta   --tokenizer_name=microsoft/codebert-base   --model_name_or_path=microsoft/codebert-base   --do_test   --do_local_explanation   --top_k_constant=10   --reasoning_method=all   --train_data_file=../data/big-vul_dataset/train.csv   --eval_data_file=../data/big-vul_dataset/val.csv   --test_data_file=../data/big-vul_dataset/test.csv   --block_size 512   --eval_batch_size 512

### RQ2 Top-10 Accuracy and IFA of CppCheck
sudo apt-get install cppcheck
cd cppcheck
python run.py

## RQ3
### Effort@20%Recall and Recall@1%LOC
cd linevul
python linevul_main.py   --model_name=12heads_linevul_model.bin   --output_dir=./saved_models   --model_type=roberta   --tokenizer_name=microsoft/codebert-base   --model_name_or_path=microsoft/codebert-base   --do_test   --do_sorting_by_line_scores   --effort_at_top_k=0.2   --top_k_recall_by_lines=0.01   --top_k_recall_by_pred_prob=0.2   --reasoning_method=all   --train_data_file=../data/big-vul_dataset/train.csv   --eval_data_file=../data/big-vul_dataset/val.csv   --test_data_file=../data/big-vul_dataset/test.csv   --block_size 512   --eval_batch_size 512

### result of Effort@20%Recall and Recall@1%LOC of CppCheck
cd cppcheck
python run.py

## ablation study
### LineVul model
see RQ1

### BPE+No Pretraining+BERT
cd linevul
cd saved_models
cd checkpoint-best-f1
gdown https://drive.google.com/uc?id=1yTe42JK_Z5ZB9MHb4eIKIMu-uqH0fE_m
cd ../../..
cd linevul
python linevul_main.py   --model_name=bpebert.bin   --output_dir=./saved_models   --model_type=roberta   --tokenizer_name=microsoft/codebert-base   --model_name_or_path=microsoft/codebert-base   --do_test   --train_data_file=../data/big-vul_dataset/train.csv   --eval_data_file=../data/big-vul_dataset/val.csv   --test_data_file=../data/big-vul_dataset/test.csv   --block_size 512   --eval_batch_size 512

### Word-Level+Pretraining(Codesearchnet)+BERT
cd linevul
cd saved_models
cd checkpoint-best-f1
gdown https://drive.google.com/uc?id=1cXeaWeBCpBuY6gPkRft2tS7SnDZrBed-
cd ../../..
cd linevul
python linevul_main.py   --model_name=WordlevelPretrainedBERT.bin   --output_dir=./saved_models   --model_type=roberta   --tokenizer_name=microsoft/codebert-base   --model_name_or_path=microsoft/codebert-base   --do_test   --train_data_file=../data/big-vul_dataset/train.csv   --eval_data_file=../data/big-vul_dataset/val.csv   --test_data_file=../data/big-vul_dataset/test.csv   --block_size 512   --eval_batch_size 512

### Word-Level+No Pretraining+BERT
cd linevul
cd saved_models
cd checkpoint-best-f1
gdown https://drive.google.com/uc?id=1yTe42JK_Z5ZB9MHb4eIKIMu-uqH0fE_m
cd ../../..
cd linevul
python linevul_main.py   --model_name=WordlevelBERT.bin   --output_dir=./saved_models   --model_type=roberta   --tokenizer_name=microsoft/codebert-base   --model_name_or_path=microsoft/codebert-base   --do_test   --train_data_file=../data/big-vul_dataset/train.csv   --eval_data_file=../data/big-vul_dataset/val.csv   --test_data_file=../data/big-vul_dataset/test.csv   --block_size 512   --eval_batch_size 512