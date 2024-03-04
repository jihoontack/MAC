from transformers import Adafactor, AutoModelForCausalLM
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import csv
import math

from utils import decode_to_clean_text, exact_match, f1_score


def gen_save(model, path):
    if getattr(model, "save_pretrained", None) is not None:
        model.save_pretrained(path)
    else:
        torch.save(model, path)


def embeddings_to_ids(embeddings, embedding_matrix, cos=True):
    """
    Convert embeddings to token IDs based on closest cosine similarity.

    Args:
    - embeddings (torch.Tensor): shape [sequence_length, embedding_dim]
    - embedding_matrix (torch.Tensor): shape [vocab_size, embedding_dim]

    Returns:
    - token_ids (torch.Tensor): shape [sequence_length]
    """
    token_ids = []

    for embedding in embeddings:
        # Calculate cosine similarity with all embeddings in the matrix
        if cos:
            similarities = F.cosine_similarity(embedding.unsqueeze(0), embedding_matrix)
        else:
            similarities = - F.pairwise_distance(embedding.unsqueeze(0), embedding_matrix)

        # Find the token ID with the highest similarity
        closest_token_id = torch.argmax(similarities).item()
        token_ids.append(closest_token_id)

    return torch.tensor(token_ids)


def qa_eval(cfg, dataloader, log_path, model, tokenizer, context_summary_bank=None, device=None,
            top_k=1, diversity_penalty=10., num_beam_groups=4, num_beams=12, amort_model=None, oracle=False):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()

    total_cnt = 0
    em_correct = 0
    avg_f1s = []
    max_f1s = []

    if context_summary_bank is not None:
        context_summary_bank = context_summary_bank.to(device)

    use_cache = True if cfg.base_model in ['Llama2_7b'] else False

    with open(log_path, 'w', newline='') as writefile:
        writer = csv.writer(writefile)
        for batch in tqdm(dataloader):

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
                if isinstance(v, list):
                    if isinstance(v[0], torch.Tensor):
                        batch[k] = [x.to(device) for x in v]

            torch.cuda.empty_cache()
            kwargs = {}
            if context_summary_bank is not None:
                assert oracle is False
                with torch.no_grad():
                    continuous_prompts = amort_model.predict_prompt_from_memory(
                        batch['gen_q_ids_amort'],
                        batch['gen_q_attn_mask_amort'],
                        context_summary_bank
                    )
                kwargs['peft_generation'] = True
                kwargs['prompts'] = continuous_prompts

            if batch['gen_q_ids'].shape[1] > 256:
                batch['gen_q_ids'] = batch['gen_q_ids'][:, -256:]
                batch['gen_q_attn_mask'] = batch['gen_q_attn_mask'][:, -256:]

            with torch.no_grad():
                outs = model.generate(
                    input_ids=batch['gen_q_ids'],
                    attention_mask=batch["gen_q_attn_mask"],
                    use_cache=use_cache,
                    max_length=batch['gen_q_ids'].shape[1] + 16,
                    num_return_sequences=top_k,
                    num_beam_groups=num_beam_groups,
                    num_beams=num_beams,
                    diversity_penalty=diversity_penalty,
                    early_stopping=True,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    **kwargs
                )
            dec = decode_to_clean_text(tokenizer, outs)
            texts = decode_to_clean_text(tokenizer, batch['gen_q_ids'])
            targets = decode_to_clean_text(tokenizer, batch['answer_ids'])

            for i in range(len(batch['gen_q_ids'])):
                total_cnt += 1
                answer = targets[i]

                predicted_answers = [dec[i * top_k + j][len(texts[i]):] for j in range(top_k)]

                answer_token = outs[i][len(batch['gen_q_ids'][i]):]
                # print('------')
                # print (f"Question {tokenizer.decode(batch['gen_q_ids'][i], skip_special_tokens=True)}")
                # print (f"Answer GT: {answer}")
                # print (f"Answer Pred: {tokenizer.decode(answer_token, skip_special_tokens=True)}")
                # print (f"Answer Pred token: {answer_token}")
                # print ('------')

                em = 0
                f1s = []
                for pred_ans in predicted_answers:
                    if exact_match(pred_ans, answer, match_length=False):
                        em = 1
                    f1s.append(f1_score(pred_ans, answer))
                em_correct += em
                writer.writerow([texts[i], answer, predicted_answers, dec[i], f1s, em])
                avg_f1s.append(np.mean(f1s))
                max_f1s.append(np.max(f1s))
        writer.writerow(['EM', em_correct, em_correct / total_cnt])
        writer.writerow(['avg_f1', np.mean(avg_f1s), np.std(avg_f1s)])
        writer.writerow(['max_f1', np.mean(max_f1s), np.std(max_f1s)])
    print('done evaluating: EM', em_correct / total_cnt)
    print('done evaluating: avg_f1', np.mean(avg_f1s), np.std(avg_f1s))
    print('done evaluating: max_f1', np.mean(max_f1s), np.std(max_f1s))
    return em_correct, em_correct / total_cnt


def context_summarization(dataloader, amort_model, train=False):
    print('Context Summarization')
    prompts = []
    amort_model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for i_step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
            if isinstance(v, list):
                if isinstance(v[0], torch.Tensor):
                    batch[k] = [x.to(device) for x in v]
        with torch.no_grad():
            prompt = amort_model.context_amortize(batch["text_ids_amort"], batch["text_attention_amort"], train=train)
        prompts.append(prompt.detach().cpu())
        # break
    return torch.cat(prompts, dim=0)

