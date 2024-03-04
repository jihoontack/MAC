# MAC
Official PyTorch implementation of "Online Adaptation of Language Models with a Memory of Amortized Contexts".

## Conda
```
conda create -n mac python=3.8 -y
conda activate mac

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121  # cu181 for cuda 11.1
pip install transformers==4.36.2 peft==0.7.1 accelerate==0.25.0 ipykernel==6.29.0 hydra-core==1.2.0 higher==0.2.1 pandas==2.0.3 datasets==2.16.1 spacy==3.7.2 Pillow==10.2.0 matplotlib==3.7.4 protobuf==4.25.2 einops==0.7.0 wandb==0.16.2 bitsandbytes==0.42.0 sentencepiece==0.1.99 deepspeed==0.13.1
```

## Prepare data

Download data to `/data` folder\
or change the data_dir in `./conf/dataset/<DATASET_NAME>.yaml`
- StreamingQA: https://drive.google.com/drive/folders/17qcGurJznFPta9Qo0z6YGtwh1fwRHm2f
- SQuAD: Automatically downloaded by huggingface datasets
- ArchivaQA: https://github.com/nathanhu0/CaMeLS

## How to run

**WANDB**: To use weight and bias (wandb) logging
- Create a wandb account and get your wandb key
- Set `wandb_key` in `./conf/config.yaml` as your wandb key
- `wandb_project` in `./conf/config.yaml` is the name of your wandb project
- `wandb_entity` in `./conf/config.yaml` is your wandb entity name
- Set `wandb_log` as false if you don't want to use wandb logging

**DATA and CACHE**: Some important paths
- `./conf/dataset/streamingqa.yaml`: dataset path
- `CACHE_DIR` in `./conf/config.yaml`: cache path for huggingface model download (e.g., GPT2, T5 model parameters and tokenizers)

**BATCH_SIZE**: Have verified that the current batch size in the config file is able to run with **2 GPUs (48GB each)**
- Actual batch size: `update_batch_size` * `grad_acc_steps`
- `update_batch_size`: batch size for 1 iteration (considering all gpus)
- `grad_acc_steps`: number of gradient accumulation steps
- batch size per gpu for 1 iteration: update_batch_size // number of gpus

Use `bf16` for mixed precision training as `fp16` does not go well with t5 (see: https://github.com/huggingface/transformers/issues/17978)

```
# train distillgpt2
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m accelerate.commands.launch --config_file ./conf/accelerate_config.yaml --num_processes=4 main.py mode=amortize_encdec_distillgpt2 dataset=streamingqa

# train gpt2-large
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m accelerate.commands.launch --config_file ./conf/accelerate_config.yaml --num_processes=4 main.py mode=amortize_encdec_gpt2large dataset=streamingqa mixed_precision=bf16 

# train gpt2-xl
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m accelerate.commands.launch --config_file ./conf/accelerate_config.yaml --num_processes=4 main.py mode=amortize_encdec_gpt2xl dataset=streamingqa mixed_precision=bf16 

# train llama2
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m accelerate.commands.launch --config_file ./conf/zero2_config.yaml --num_processes=4 main.py mode=amortize_encdec_llama2_7b dataset=streamingqa mixed_precision=bf16 quant_type=nf4 llama_cache_dir=<LLAMA_PATH>
```

## Evaluation code
```
# Evaluate on StreamingQA
CUDA_VISIBLE_DEVICES=0 python eval.py mode_eval=amortize_encdec_distillgpt2 dataset=streamingqa load_path=<LOAD_PATH>
```

