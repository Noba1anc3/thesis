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

