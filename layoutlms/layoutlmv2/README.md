## Fine-tuning Example on FUNSD

### Installation

Please refer to [layoutlmft](../layoutlmft/README.md)

### Command

```
cd layoutlmft
python -m torch.distributed.launch --nproc_per_node=4 examples/run_funsd.py \
        --model_name_or_path microsoft/layoutlmv2-base-uncased \
        --output_dir /tmp/test-ner \
        --do_train \
        --do_predict \
        --max_steps 1000 \
        --warmup_ratio 0.1 \
        --fp16
```



python examples/run_funsd.py \
        --model_name_or_path microsoft/layoutlmv2-base-uncased \
        --output_dir output \
        --do_train \
        --do_eval \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --eval_steps 200 \
        --logging_steps 100 \
        --save_steps 100 \
        --num_train_epochs 50.0 \

python examples/run_funsd.py \
        --model_name_or_path microsoft/layoutlmv2-base-uncased \
        --output_dir ../../../../../../drive/'My Drive'/output \
        --do_train \
        --do_eval \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --train_batch_size 8 \
        --eval_batch_size 8 \
        --eval_steps 200 \
        --logging_steps 100 \
        --save_steps 100 \
        --num_train_epochs 50.0 \
        --overwrite_output_dir \
        --warmup_ratio 0.1


max_steps 1000
max_train_samples
max_val_samples
label_all_tokens False
