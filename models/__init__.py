import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, LlamaForCausalLM
from models.peft_wrapper import BaseModelPeftWrapper
from hydra.utils import to_absolute_path

from models.t5_wrapper import T5ForwardWrapper
from models.amortized_encdec import AmortEncDecAggregateWrapper


def prepare_prompt_learning_config(config, model_config):
    if config.token_dim is None:
        if hasattr(model_config, "hidden_size"):
            token_dim = model_config.hidden_size
        elif hasattr(model_config, "n_embd"):
            token_dim = model_config.n_embd
        elif hasattr(model_config, "d_model"):
            token_dim = model_config.d_model
        else:
            raise ValueError("Please specify `token_dim` in `config`")
    else:
        token_dim = config.token_dim

    return token_dim


def get_base_model(cfg, accelerator=None):
    kwargs = {}
    base_model_name = cfg.base_model
    if cfg.base_model in ['Llama2_7b'] or cfg.quant_type is not None:
        if cfg.quant_type == 'nf4':
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,  # for T5
                bnb_4bit_quant_type='nf4',  # Quantization type (fp4 or nf4)
                bnb_4bit_use_double_quant=False,
            )
        elif cfg.quant_type == 'int8':
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            raise ValueError(f"quant_type {cfg.quant_type} not supported")
        kwargs['quantization_config'] = bnb_config
        kwargs['torch_dtype'] = torch.float32

    if cfg.base_model == 'Llama2_7b':
        if cfg.load_from_hf:
            base_lm = LlamaForCausalLM.from_pretrained(cfg.base_model_state_dict_path, **kwargs)
        else:
            base_lm = LlamaForCausalLM.from_pretrained(cfg.llama_cache_dir, **kwargs)
    else:
        base_lm = AutoModelForCausalLM.from_pretrained(
            base_model_name, cache_dir=cfg.CACHE_DIR, **kwargs
        )

    if cfg.base_model_state_dict_path is not None and not cfg.load_from_hf:
        if accelerator is not None:
            map_location = accelerator.device
        else:
            map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        unexpected = base_lm.load_state_dict(torch.load(cfg.base_model_state_dict_path, map_location), strict=False)
        print (f"Unexpected keys: {unexpected.unexpected_keys}")

    if cfg.token_dim is None:
        cfg.token_dim = prepare_prompt_learning_config(cfg, base_lm.config)

    for param in base_lm.parameters():  # free the base model
        param.requires_grad = False
    base_lm = BaseModelPeftWrapper(base_lm, cfg)

    base_lm.train()
    return base_lm


def get_amortize_encdec_model(cfg, base_lm, tokenizer=None, tokenizer_amort=None):
    if 't5' in cfg.pretrained_model_amort:
        kwargs = {}
        if cfg.quant_type is not None:
            if cfg.quant_type == 'nf4':
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,  # for T5
                    bnb_4bit_quant_type='nf4',  # Quantization type (fp4 or nf4)
                    bnb_4bit_use_double_quant=False,
                )
            elif cfg.quant_type == 'int8':
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            else:
                raise ValueError(f"quant_type {cfg.quant_type} not supported")
            kwargs['quantization_config'] = bnb_config
            kwargs['torch_dtype'] = torch.float32

        ## Load encoder-decoder model for amortization ##
        t5model = T5ForwardWrapper.from_pretrained(
            cfg.pretrained_model_amort, cache_dir=cfg.CACHE_DIR, **kwargs
        )
        learnable_prompts = torch.randn(1, cfg.num_virtual_tokens, t5model.encoder.config.hidden_size) * .02
        t5model.learnable_prompts = nn.Parameter(learnable_prompts)
        t5model.num_virtual_tokens = cfg.num_virtual_tokens
        if hasattr(t5model, 'lm_head'): del t5model.lm_head

        #################################################

        ## Load question encoder ##
        question_model_name = cfg.pretrained_model_amort if cfg.question_model_amort is None else cfg.question_model_amort
        question_encoder = T5ForwardWrapper.from_pretrained(
            question_model_name, cache_dir=cfg.CACHE_DIR, **kwargs
        )
        learnable_prompts = torch.randn(1, cfg.num_virtual_tokens, question_encoder.encoder.config.hidden_size) * .02
        question_encoder.learnable_prompts = nn.Parameter(learnable_prompts)
        question_encoder.num_virtual_tokens = cfg.num_virtual_tokens
        if hasattr(question_encoder, 'lm_head'): del question_encoder.lm_head
        ############################

        # Wrap into amortized model
        amort_model = AmortEncDecAggregateWrapper(
            cfg, base_lm=base_lm, enc_decoder=t5model, question_encoder=question_encoder
        )
        amort_model.tokenizer = tokenizer
        amort_model.tokenizer_amort = tokenizer_amort
    else:
        raise NameError('Unknown model type')

    if cfg.load_path is not None:
        amort_model.load(target_path=to_absolute_path(cfg.load_path))

    amort_model = amort_model
    amort_model.freeze_param()

    return amort_model
