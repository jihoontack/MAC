import warnings
from typing import Optional

import packaging.version
import torch
import torch.nn as nn
import transformers
from transformers.cache_utils import DynamicCache


def _get_batch_size(input_ids: Optional[torch.Tensor], inputs_embeds: Optional[torch.Tensor]) -> int:
    """Get the batch size based on either input_ids or input_embeds

    Raises an ValueError if both are None.

    """
    if (input_ids is None) and (inputs_embeds is None):
        raise ValueError("You have to provide either input_ids or inputs_embeds")

    if input_ids is not None:
        batch_size = input_ids.shape[0]
    else:
        batch_size = inputs_embeds.shape[0]
    return batch_size


class BaseModelPeftWrapper(nn.Module):
    def __init__(self, base_lm, config_peft):
        super().__init__()
        self.config_peft = config_peft
        self.base_lm = base_lm
        self.config = base_lm.config
        self.base_model_prepare_inputs_for_generation = self.base_lm.prepare_inputs_for_generation
        self.llama = True if config_peft.base_model in ["Llama2_7b"] else False
        self.num_virtual_tokens = self.config_peft.layer_num_virtual_tokens

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            prompts=None,
            **kwargs,
    ):

        # if there is no prompts in kwargs, use the default forward
        if prompts is None:
            return self.base_lm.forward(
                input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, labels=labels,
                output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                return_dict=return_dict, **kwargs
            )

        batch_size = _get_batch_size(input_ids, inputs_embeds)
        if attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(batch_size, self.num_virtual_tokens).to(attention_mask.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        if kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        if kwargs.get("token_type_ids", None) is not None:
            warnings.warn("Token type ids are not supported for parameter efficient tuning. Ignoring token type ids")
            kwargs["token_type_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        past_key_values = prompts
        # use the forward function of the AutoModelForCausalLM
        return self.base_lm.forward(
            input_ids=input_ids, inputs_embeds=inputs_embeds, past_key_values=past_key_values, **kwargs
        )

    def generate(self, peft_generation=False, **kwargs):
        if peft_generation:
            self.base_lm.prepare_inputs_for_generation = self.prepare_inputs_for_generation_peft
        else:
            self.base_lm.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation

        if 'prompts' in kwargs:
            # repeat prompts by num_beams
            if kwargs["num_beams"] > 1:
                # prompts is a tuple
                prompts = []
                for prompt in kwargs['prompts']:
                    prompts.append(prompt.repeat(1, kwargs["num_beams"], 1, 1, 1))
                kwargs['prompts'] = tuple(prompts)

        kwargs['prompts'] = DynamicCache.from_legacy_cache(kwargs['prompts'])

        try:
            outputs = self.base_lm.generate(**kwargs)
        except:
            self.base_lm.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            raise
        else:
            self.base_lm.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            return outputs

    def prepare_inputs_for_generation_peft(self, *args, prompts: torch.Tensor = None, **kwargs):
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)

        # https://github.com/huggingface/transformers/pull/26681/ introduced new cache format
        # for some architectures which requires a special fix for prompt tuning etc.
        # TODO: starting with transformers 4.37, all architectures should support caching.
        uses_transformers_4_37 = packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.37.0")
        uses_transformers_4_36 = packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.36.0")
        transformers_new_cache_archs = ["llama", "mistral", "persimmon", "phi"]
        uses_cache = uses_transformers_4_37 or (
                uses_transformers_4_36 and self.base_lm.config.model_type in transformers_new_cache_archs
        )

        if uses_cache and (model_kwargs["past_key_values"] is not None):
            # change in the logic of `prepare_inputs_for_generation` makes the below code necessary
            # In prompt learning methods, past key values are longer when compared to the `input_ids`.
            # As such only consider the last input ids in the autogressive generation phase.
            if model_kwargs["past_key_values"][0][0].shape[-2] >= model_kwargs["input_ids"].shape[1]:
                model_kwargs["input_ids"] = model_kwargs["input_ids"][:, -1:]

        if model_kwargs.get("attention_mask", None) is not None:
            size = model_kwargs["input_ids"].shape[0], self.num_virtual_tokens
            prefix_attention_mask = torch.ones(size).to(model_kwargs["input_ids"].device)
            model_kwargs["attention_mask"] = torch.cat(
                (prefix_attention_mask, model_kwargs["attention_mask"]), dim=1
            )

        if model_kwargs.get("position_ids", None) is not None:
            model_kwargs["position_ids"] = None

        if kwargs.get("token_type_ids", None) is not None:
            kwargs["token_type_ids"] = None

        if model_kwargs["past_key_values"] is None:
            model_kwargs["past_key_values"] = prompts
        else:
            if model_kwargs["past_key_values"] is None:
                # if there is word_embeddings, use it, otherwise use wte
                if hasattr(self.base_lm, "word_embeddings"):
                    inputs_embeds = self.base_lm.word_embeddings(model_kwargs["input_ids"])
                else:
                    inputs_embeds = self.base_lm.transformer.wte(model_kwargs["input_ids"])
                model_kwargs["inputs_embeds"] = torch.cat((prompts, inputs_embeds), dim=1)
                model_kwargs["input_ids"] = None

        return model_kwargs
