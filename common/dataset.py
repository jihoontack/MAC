import copy
import json

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from transformers import AutoTokenizer
from datasets import load_dataset

from utils import shuffle_groups, return_k_unique, cycle
import ast


class RangeSampler(Sampler):
    def __init__(self, start_index, end_index):
        self.start_index = start_index
        self.end_index = end_index
        super().__init__(range(start_index, end_index))

    def __len__(self):
        return self.end_index - self.start_index

    def __iter__(self):
        return iter(range(self.start_index, self.end_index))


class TextAndQuestionDataset(Dataset):
    def __init__(self, max_text_len=1024, max_question_len=128, device=None, loc=False, qa_only=False,
                 qa_for_generation=False, max_answer_len=24, tokenizer='gpt2', prompt_samples=-1,
                 pad_qa_for_gen=True, include_eos=True, tokenizer_amort=None, num_virtual_tokens=20,
                 cache_dir=None):
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, cache_dir=cache_dir)
        else:
            self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_text_len = max_text_len
        self.qa_for_generation = qa_for_generation
        self.qa_only = qa_only
        self.max_question_len = max_question_len
        self.max_answer_len = max_answer_len
        self.loc = loc
        self.prompt_samples = prompt_samples
        self.pad_qa_for_gen = pad_qa_for_gen
        self.include_eos = include_eos
        self.tokenizer_amort = tokenizer_amort
        self.min_question_len = max_answer_len

    def __len__(self):
        raise NotImplementedError("Subclasses must implement __len__")

    def get_qa(self, idx):
        # return text corresponding to a question and answer pair at index idx
        # we expect answer to not have a space at the beginning
        raise NotImplementedError("Subclasses must implement get_qa")

    def get_text(self, idx):
        # return text corresponding to the passage with information at index idx
        raise NotImplementedError("Subclasses must implement get_text")

    def get_retreive_document(self, idx):
        doc_idx = self.retrieval_idx[idx]
        return self.get_text(doc_idx[0])

    def tok_qa_for_training(self, idx, tokenizer=None):
        question, answer = self.get_qa(idx)
        tokenizer = tokenizer if tokenizer is not None else self.tokenizer

        if self.include_eos:
            answer = answer + tokenizer.eos_token
        tok_answer = tokenizer(' ' + answer, return_tensors="pt")
        # in order to create a mask of target_ids which only computes loss on the questions answer,
        # we tokenize the question and answer separately then concatenate
        tok_question = tokenizer(question, return_tensors="pt")
        qa_ids = torch.cat([tok_question['input_ids'], (tok_answer['input_ids'])], 1)

        if qa_ids.shape[1] > self.max_question_len + self.max_answer_len:
            print(
                f'total question len {qa_ids.shape[1]} excedes max_question len f{self.max_question_len}. Truncating:')
            print(idx)
            num_to_truncate = qa_ids.shape[1] - self.max_question_len
            qa_ids = qa_ids[:, num_to_truncate:]
            tok_question['input_ids'] = tok_question['input_ids'][:, num_to_truncate:]
            tok_question['attention_mask'] = tok_question['attention_mask'][:, num_to_truncate:]

        n_pad = self.max_question_len - qa_ids.shape[1]
        qa_attention = torch.cat([tok_question['attention_mask'], (tok_answer['attention_mask'])], 1)
        qa_target_ids = qa_ids.clone()
        qa_target_ids[:, :tok_question['input_ids'].shape[1]] = -100
        qa_ids = torch.nn.functional.pad(qa_ids, (0, n_pad), value=tokenizer.pad_token_id)
        qa_attention = torch.nn.functional.pad(qa_attention, (0, n_pad), value=0)
        qa_target_ids = torch.nn.functional.pad(qa_target_ids, (0, n_pad), value=-100)

        return qa_ids, qa_attention, qa_target_ids

    def tok_qa_for_generation(self, idx, tokenizer=None):
        pad = True
        if tokenizer is None:
            tokenizer = self.tokenizer
            pad = False

        tokenizer = tokenizer if tokenizer is not None else self.tokenizer
        question, answer = self.get_qa(idx)

        if self.include_eos:
            answer = answer + tokenizer.eos_token
        if self.pad_qa_for_gen:
            tokenizer.padding_side = 'left'
            tok_question = tokenizer(question, max_length=self.max_question_len - self.max_answer_len,
                                     padding='max_length', truncation=True, return_tensors="pt")
            tokenizer.padding_side = 'right'
        else:
            tok_question = tokenizer(question, return_tensors="pt")
            if pad:
                tok_question['input_ids'] = torch.nn.functional.pad(
                    tok_question['input_ids'], (0, self.max_question_len), value=tokenizer.pad_token_id
                )
                tok_question['attention_mask'] = torch.nn.functional.pad(
                    tok_question['attention_mask'], (0, self.max_question_len), value=0
                )
        tok_answer = tokenizer(' ' + answer, max_length=self.max_answer_len, padding='max_length',
                               return_tensors="pt", truncation=True)
        return {f'gen_q_ids': tok_question['input_ids'].squeeze(),
                f'gen_q_attn_mask': tok_question['attention_mask'].squeeze(),
                f'question_text': question,
                f'answer_text': answer,
                f'answer_ids': tok_answer['input_ids'].squeeze(),
                f'answer_mask': tok_answer['attention_mask'].squeeze()}

    def __getitem__(self, idx):
        qa_ids, qa_attention, qa_target_ids = self.tok_qa_for_training(idx)
        if self.tokenizer_amort is not None:
            qa_amort = self.tok_qa_for_training(idx, tokenizer=self.tokenizer_amort)
        if self.loc:
            return_dic = {'loc_ids': qa_ids.squeeze(),
                          'loc_attention': qa_attention.squeeze(),
                          'loc_mask': torch.roll(qa_target_ids.squeeze() != -100, -1, 0)}
            if self.tokenizer_amort is not None:
                return_dic.update({'loc_ids_amort': qa_amort[0].squeeze()})
            return return_dic
        if self.qa_only:
            return_dic = {'idx': torch.tensor(idx),
                          'qa_ids': qa_ids.squeeze(),
                          'qa_attention': qa_attention.squeeze(),
                          'qa_target_ids': qa_target_ids.squeeze()}
        else:
            text = self.tokenizer(self.get_text(idx), max_length=self.max_text_len, padding='max_length',
                                  truncation=True, return_tensors="pt")
            return_dic = {'idx': torch.tensor(idx),
                          'text_ids': text['input_ids'].squeeze(),
                          'text_attention': text['attention_mask'].squeeze(),
                          'qa_ids': qa_ids.squeeze(),
                          'qa_attention': qa_attention.squeeze(),
                          'qa_target_ids': qa_target_ids.squeeze()}
            if self.tokenizer_amort is not None:
                text_amort = self.tokenizer_amort(self.get_text(idx), max_length=self.max_text_len,
                                                  padding='max_length', truncation=True, return_tensors="pt")
                return_dic.update({'text_ids_amort': text_amort['input_ids'].squeeze(),
                                   'text_attention_amort': text_amort['attention_mask'].squeeze()})

        if self.tokenizer_amort is not None:
            qa_amort = self.tok_qa_for_generation(
                idx, tokenizer=self.tokenizer_amort
            )
            return_dic.update({'gen_q_ids_amort': qa_amort['gen_q_ids'],
                               'gen_q_attn_mask_amort': qa_amort['gen_q_attn_mask']})
        if self.qa_for_generation:
            return_dic.update(self.tok_qa_for_generation(idx))

        return return_dic


class StreamingQADataset(TextAndQuestionDataset):
    def __init__(self, csv_path, downsample_to=-1, **kwargs):
        self.csv_path = csv_path
        if downsample_to != -1:
            self.data_frame = pd.read_csv('./conf/dataset/streaming_test.csv')
            # self.data_frame = self.data_frame.sample(downsample_to)
        else:
            self.data_frame = pd.read_csv(csv_path)
            self.data_frame = self.data_frame.sample(frac=1)
        super().__init__(**kwargs)

    def __len__(self):
        return len(self.data_frame)

    def get_qa(self, idx):
        row = self.data_frame.iloc[idx]
        answers = row['answers'].split("\\")
        answer = min(answers, key=len)
        question = row['question'].strip()
        return question, answer

    def get_text(self, idx):
        return self.data_frame.iloc[idx]['text']


class SquadDataset(TextAndQuestionDataset):

    def __init__(self, split, start_idx=0, end_idx=-1, shuffle_by='title', downsample_to=-1, downsample_by='context',
                 cache_dir=None, **kwargs):
        squad_ds = load_dataset('squad', split=split, cache_dir=cache_dir)
        if end_idx == -1:
            end_idx = len(squad_ds)
        squad_ds = squad_ds.select(list(range(start_idx, end_idx)))
        if downsample_to > 0:
            self.data_frame = pd.read_csv('./conf/dataset/squad_test.csv')
            self.data_frame['answers'] = self.data_frame['answers'].apply(ast.literal_eval)
            # self.data_frame = return_k_unique(self.data_frame, downsample_to, downsample_by)
        else:
            self.data_frame = pd.DataFrame(squad_ds)
            self.data_frame = shuffle_groups(self.data_frame, shuffle_by)
        super().__init__(**kwargs)

    def __len__(self):
        return len(self.data_frame)

    def get_qa(self, idx):
        question = self.data_frame.iloc[idx]['question'].strip()
        answer = min(self.data_frame.iloc[idx]['answers']['text'], key=len).strip()
        if answer[0].islower():
            answer = answer[0].upper() + answer[1:]
        return question, answer

    def get_text(self, idx):
        return self.data_frame.iloc[idx]['context']

    def get_deduplicated_dataset(self):
        new_squad_ds = copy.deepcopy(self)
        new_squad_ds.data_frame = self.data_frame.drop_duplicates(subset=['context'])
        return new_squad_ds


class ArchivalQADataset(TextAndQuestionDataset):
    def __init__(self, csv_path, full_passage=False, shuffle_by='doc_id', downsample_to=-1,
                 downsample_by='ans_paragraph', **kwargs):
        self.csv_path = csv_path
        self.full_passage = full_passage
        if downsample_to > 0:
            self.data_frame = pd.read_csv('./conf/dataset/archival_test.csv')
            # self.data_frame = return_k_unique(self.data_frame, downsample_to, downsample_by)
        else:
            self.data_frame = pd.read_csv(csv_path)
            # we sort pre shuffle to make sure that for any given doc_id, the examples are in increasing order of para_num
            self.data_frame.sort_values('para_num', kind='stable', inplace=True)
            self.data_frame = shuffle_groups(self.data_frame, shuffle_by)
        super().__init__(**kwargs)

    def __len__(self):
        return len(self.data_frame)

    def get_qa(self, idx):
        row = self.data_frame.iloc[idx]
        answer = row['answer']
        if answer[0].islower():
            answer = answer[0].upper() + answer[1:]
        question = row['question'].strip()
        return question, answer

    def get_text(self, idx):
        if self.full_passage:
            return self.data_frame.iloc[idx]['ans_text']
        return self.data_frame.iloc[idx]['ans_paragraph']

    def get_deduplicated_dataset(self):
        new_squad_ds = copy.deepcopy(self)
        if self.full_passage:
            new_squad_ds.data_frame = self.data_frame.drop_duplicates(subset=['ans_text'])
        else:
            new_squad_ds.data_frame = self.data_frame.drop_duplicates(subset=['ans_paragraph'])
        return new_squad_ds


class WebTextDataset(Dataset):
    def __init__(self, csv_path, max_text_len=1024, device_=None, loc=False, tokenizer='gpt2',
                 tokenizer_amort=None, cache_dir=None):
        self.csv_path = csv_path
        self.device = device_ if device_ is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_frame = pd.read_csv(csv_path)
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, cache_dir=cache_dir)
        else:
            self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer_amort = tokenizer_amort
        if self.tokenizer_amort is not None:
            self.tokenizer_amort.pad_token = self.tokenizer_amort.eos_token
        self.max_text_len = max_text_len
        self.loc = loc

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        text = self.tokenizer(row['raw_text'], max_length=self.max_text_len, padding='max_length',
                              truncation=True, return_tensors="pt")
        if self.tokenizer_amort is None:
            if self.loc:
                return {'loc_ids': text['input_ids'].squeeze(),
                        'loc_attention': text['attention_mask'].squeeze(),
                        'loc_mask': text['attention_mask'].squeeze()}
            else:
                return {'input_ids': text['input_ids'].squeeze(),
                        'attention_mask': text['attention_mask'].squeeze()}
        else:
            text_amort = self.tokenizer_amort(row['raw_text'], max_length=self.max_text_len, padding='max_length',
                                              truncation=True, return_tensors="pt")
            if self.loc:
                return {'loc_ids': text['input_ids'].squeeze(),
                        'loc_attention': text['attention_mask'].squeeze(),
                        'loc_mask': text['attention_mask'].squeeze(),
                        'loc_ids_amort': text_amort['input_ids'].squeeze()}
            else:
                return {'input_ids': text['input_ids'].squeeze(),
                        'attention_mask': text['attention_mask'].squeeze(),
                        'input_ids_amort': text_amort['input_ids'].squeeze()}


def get_dataloader(cfg, tokenizer, tokenizer_amort=None):
    max_text_len = 1024
    num_virtual_tokens = 0
    if 'amort' in cfg.mode:
        num_virtual_tokens = cfg.layer_num_virtual_tokens
        max_text_len = 1024 - num_virtual_tokens

    kwargs_train = {}
    kwargs_val = {}

    if cfg.dataset == 'streamingqa':
        train_dataset = StreamingQADataset(
            cfg.train_path, tokenizer=tokenizer,
            max_text_len=max_text_len, tokenizer_amort=tokenizer_amort,
            cache_dir=cfg.CACHE_DIR, **kwargs_train
        )
        val_dataset = StreamingQADataset(
            cfg.val_path, tokenizer=tokenizer,
            max_text_len=max_text_len, tokenizer_amort=tokenizer_amort,
            cache_dir=cfg.CACHE_DIR, **kwargs_val,
        )
        val_dataset_gen = StreamingQADataset(
            cfg.val_path, tokenizer=tokenizer, qa_for_generation=True,
            pad_qa_for_gen=False, max_text_len=max_text_len,
            tokenizer_amort=tokenizer_amort, cache_dir=cfg.CACHE_DIR,
            num_virtual_tokens=num_virtual_tokens
        )
    elif cfg.dataset == 'squad':
        train_dataset = SquadDataset(
            cfg.train_split, cfg.train_start_idx, cfg.train_end_idx, tokenizer=tokenizer,
            max_text_len=max_text_len, tokenizer_amort=tokenizer_amort,
            cache_dir=cfg.CACHE_DIR, **kwargs_train
        )
        val_dataset = SquadDataset(
            cfg.val_split, cfg.val_start_idx, cfg.val_end_idx, tokenizer=tokenizer,
            max_text_len=max_text_len, tokenizer_amort=tokenizer_amort,
            cache_dir=cfg.CACHE_DIR, **kwargs_val
        )
        val_dataset_gen = SquadDataset(
            cfg.val_split, cfg.val_start_idx, cfg.val_end_idx, tokenizer=tokenizer,
            max_text_len=max_text_len, tokenizer_amort=tokenizer_amort, qa_for_generation=True,
            pad_qa_for_gen=False, cache_dir=cfg.CACHE_DIR, **kwargs_val,
        )
    elif cfg.dataset == 'archivalqa':
        train_dataset = ArchivalQADataset(
            cfg.train_path, tokenizer=tokenizer, full_passage=cfg.full_passage,
            max_text_len=max_text_len, tokenizer_amort=tokenizer_amort,
            cache_dir=cfg.CACHE_DIR, **kwargs_train
        )
        val_dataset = ArchivalQADataset(
            cfg.val_path, tokenizer=tokenizer, full_passage=cfg.full_passage,
            max_text_len=max_text_len, tokenizer_amort=tokenizer_amort,
            cache_dir=cfg.CACHE_DIR, **kwargs_val
        )
        val_dataset_gen = ArchivalQADataset(
            cfg.val_path, tokenizer=tokenizer, full_passage=cfg.full_passage,
            max_text_len=max_text_len, tokenizer_amort=tokenizer_amort, qa_for_generation=True,
            pad_qa_for_gen=False, cache_dir=cfg.CACHE_DIR, **kwargs_val
        )
    else:
        raise NotImplementedError(f"Dataset {cfg.dataset} not implemented")

    if cfg.lift_ratio != 1.0:
        batch_size = int((cfg.update_batch_size // cfg.world_size) / cfg.lift_ratio)
        cfg.n_epochs = int(cfg.n_epochs / cfg.lift_ratio)
        cfg.val_steps = int(cfg.val_steps / cfg.lift_ratio)
    else:
        batch_size = cfg.update_batch_size // cfg.world_size

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size, drop_last=True
    )
    val_dataloader = DataLoader(
        val_dataset, shuffle=False, batch_size=cfg.update_val_batch_size//cfg.world_size
    )
    val_gen_dataloader = DataLoader(
        val_dataset_gen, shuffle=False, batch_size=1
    )

    return train_dataloader, val_dataloader, val_gen_dataloader


def get_loc_dataloader(cfg, tokenizer, tokenizer_amort=None):
    max_text_len = 1024

    train_dataset = WebTextDataset(csv_path=cfg.web_text_csv, loc=True,
                                   tokenizer=tokenizer, max_text_len=max_text_len,
                                   tokenizer_amort=tokenizer_amort, cache_dir=cfg.CACHE_DIR)
    val_dataset = WebTextDataset(csv_path=cfg.web_text_val_csv, loc=True,
                                 tokenizer=tokenizer, max_text_len=max_text_len,
                                 tokenizer_amort=tokenizer_amort, cache_dir=cfg.CACHE_DIR)

    loc_train_loader = DataLoader(train_dataset, batch_size=cfg.loc_batch_size, shuffle=True, drop_last=True)
    loc_val_loader = DataLoader(val_dataset, batch_size=cfg.loc_batch_size, shuffle=False, drop_last=True)
    return cycle(loc_train_loader), cycle(loc_val_loader)


def get_eval_dataloader(cfg, tokenizer, tokenizer_amort=None):
    max_text_len = 1024
    num_virtual_tokens = 0
    if 'amort' in cfg.mode_eval:
        num_virtual_tokens = cfg.layer_num_virtual_tokens
        max_text_len = 1024 - num_virtual_tokens

    if cfg.dataset == 'streamingqa':
        train_dataset = StreamingQADataset(cfg.test_path, tokenizer=tokenizer, qa_for_generation=True,
                                          pad_qa_for_gen=(cfg.batch_size != 1),
                                          downsample_to=cfg.downsample_to, max_text_len=max_text_len,
                                          tokenizer_amort=tokenizer_amort, num_virtual_tokens=num_virtual_tokens)
        test_dataset = StreamingQADataset(cfg.test_path, tokenizer=tokenizer, qa_for_generation=True,
                                          pad_qa_for_gen=False,
                                          downsample_to=cfg.downsample_to, max_text_len=max_text_len,
                                          tokenizer_amort=tokenizer_amort, num_virtual_tokens=num_virtual_tokens)
    elif cfg.dataset == 'squad':
        train_dataset = SquadDataset(cfg.test_split, cfg.test_start_idx, cfg.test_end_idx, tokenizer=tokenizer,
                                    qa_for_generation=True, pad_qa_for_gen=(cfg.batch_size != 1),
                                    downsample_to=cfg.downsample_to, max_text_len=max_text_len,
                                    tokenizer_amort=tokenizer_amort, num_virtual_tokens=num_virtual_tokens)
        test_dataset = SquadDataset(cfg.test_split, cfg.test_start_idx, cfg.test_end_idx, tokenizer=tokenizer,
                                    qa_for_generation=True, pad_qa_for_gen=False,
                                    downsample_to=cfg.downsample_to, max_text_len=max_text_len,
                                    tokenizer_amort=tokenizer_amort, num_virtual_tokens=num_virtual_tokens)
    elif cfg.dataset == 'archivalqa':
        train_dataset = ArchivalQADataset(cfg.test_path, tokenizer=tokenizer, qa_for_generation=True,
                                         pad_qa_for_gen=(cfg.batch_size != 1), downsample_to=cfg.downsample_to,
                                         full_passage=cfg.full_passage, max_text_len=max_text_len,
                                         tokenizer_amort=tokenizer_amort, num_virtual_tokens=num_virtual_tokens)
        test_dataset = ArchivalQADataset(cfg.test_path, tokenizer=tokenizer, qa_for_generation=True,
                                         pad_qa_for_gen=False, downsample_to=cfg.downsample_to,
                                         full_passage=cfg.full_passage, max_text_len=max_text_len,
                                         tokenizer_amort=tokenizer_amort, num_virtual_tokens=num_virtual_tokens)
    else:
        print(f'dataset [{cfg.dataset}] not supported for evaluation')
        raise NotImplementedError

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.generation_batch_size, shuffle=False)

    return train_dataloader, test_dataloader
