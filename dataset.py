import os
import json
from typing import *


import torch
from torch.utils.data import IterableDataset, Dataset
from tqdm import tqdm

from transformers.tokenization_utils import PreTrainedTokenizer
import copy


def load_single_file(data_file):
    with open(data_file)as f:
        lines = f.readlines()
    return [json.loads(l) for l in lines]

def load_raw_data(data_file):
    raw_dataset = []
    if isinstance(data_file, str):
        raw_dataset += load_single_file(data_file)
    elif isinstance(data_file, list):
        for f_ in data_file:
            raw_dataset += load_single_file(f_)
    return raw_dataset
    
IGNORE_INDEX=-100


def collator(tokenizer, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
    input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
    return dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class PromptIterableDataset(IterableDataset):
    def __init__(self,
                 raw_dataset: Union[Dataset, List],
                 sep: List = ["EOS", "\n"],
                 tokenizer: PreTrainedTokenizer = None,
                 max_seq_length: Optional[int] = 512,
                 teacher_forcing: Optional[bool] = True,
                 truncate_method: Optional[str] = "tail",
                ):
        assert hasattr(raw_dataset, "__iter__"), f"The dataset must have __iter__ method. dataset is {raw_dataset}"
        assert hasattr(raw_dataset, "__len__"), f"The dataset must have __len__ method. dataset is {raw_dataset}"
        self.raw_dataset = raw_dataset
        self.sep = sep
        self._end_token = None
        self.start_token = self.sep[-1]
        self.teacher_forcing = teacher_forcing
        assert self.teacher_forcing, print("must use teacher forcing")

        self.tokenizer = tokenizer
        self.truncate_method = truncate_method
        self.max_seq_length = max_seq_length
        assert self.truncate_method == "tail", print("only tail truncate support")
    

    
    @property
    def end_token(self):
        if self._end_token is not None:
            return self._end_token
        end_token = self.sep[0]
        if end_token == "EOS":
            self._end_token = self.tokenizer.eos_token
        else:
            self._end_token = end_token
        return self._end_token

    def tokenize_example(self, example):
        end_token = self.end_token
        # print(example)
        tags = [i for _ in range(len(example["data"])//2) for i in ["User", "Assistant"]]
        labels = []
        tokenized_ids = []
        for i, c in enumerate(example["data"]):
            c_new = tags[i] + ": " + c + end_token
            if i % 2 == 1:
                # model
                c_input = self.start_token + tags[i] + ": "
                tokenized = self.tokenizer(c_input, add_special_tokens=False)
                tokenized_ids += tokenized["input_ids"]
                labels += [IGNORE_INDEX] * len(tokenized["input_ids"])

                c_generate = c + end_token
                tokenized = self.tokenizer(c_generate, add_special_tokens=False)
                tokenized_ids += tokenized["input_ids"]
                labels += tokenized["input_ids"]

            else:
                # user
                if i == 0:
                    # no start token
                    c_new = self.tokenizer.bos_token + tags[i] + ": " + c + end_token
                else:
                    c_new = self.start_token + tags[i] + ": " + c + end_token
                tokenized = self.tokenizer(c_new, add_special_tokens=False)
                tokenized_ids += tokenized["input_ids"]
                labels += [IGNORE_INDEX] * len(tokenized["input_ids"])

        assert len(tokenized_ids) == len(labels)
        return {"input_ids": torch.LongTensor(tokenized_ids + (self.max_seq_length - len(tokenized_ids))*[tokenizer.pad_token_id]), "labels": torch.LongTensor(labels + (self.max_seq_length - len(tokenized_ids))*[-100])}

    def truncate(self, tokenized_example):
        old_len = len(tokenized_example["input_ids"])
        if old_len > self.max_seq_length:
            for k in tokenized_example:
                tokenized_example[k] = tokenized_example[k][:-(old_len - self.max_seq_length)]

        return tokenized_example


    def __iter__(self):
        for example in self.raw_dataset:
            tokenized_example = self.tokenize_example(example)
            tokenized_example = self.truncate(tokenized_example)
            yield tokenized_example

    def __len__(self):
        return len(self.raw_dataset)

class TextDataset(Dataset):
    def __init__(self, training_data, training_label, pad_token_id):
        self.training_data = training_data
        self.training_label = training_label
        self.pad_token_id = pad_token_id
        assert self.training_data.shape == self.training_label.shape
        
    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, idx):
        input_ids = self.training_data[idx]
        return {"input_ids": input_ids, "labels": self.training_label[idx], "attention_mask": input_ids.ne(self.pad_token_id)}
if __name__ == "__main__":
    from datasets import load_dataset
    from transformers import AutoTokenizer, LlamaTokenizer
    from torch.utils.data import DataLoader, TensorDataset, random_split
    TEMPLATE = "{} Assistant:"
    raw_dataset = load_dataset("stingning/ultrachat")["train"]
    from transformers import AutoTokenizer
    # Choose a tokenizer (e.g., LLaMA 2 or any other model)
    tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b-hf")
    # print(raw_dataset["train"])
    dataset = PromptIterableDataset(raw_dataset, tokenizer=tokenizer, max_seq_length=2048, teacher_forcing=True)
    labels = []
    input_ids = []
    cnt = 0
    for data in tqdm(dataset):
        # print(data["input_ids"].shape)
        # print(tokenizer.decode(data["input_ids"][:1000]))
        
        labels.append(data['labels'].reshape(1, -1))
        input_ids.append(data['input_ids'].reshape(1, -1))
        cnt += 1
        # if cnt >= 100: 
        #     break

    dataset = TensorDataset(torch.cat(input_ids), torch.cat(labels))
    train_size = int(0.7 * len(dataset))
    eval_size = len(dataset) - train_size

    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    # print(eval_dataset.dataset)
    torch.save(train_dataset, "dataset/mamba-ultrachat/train_dataset.pt")
    torch.save(eval_dataset, "dataset/mamba-ultrachat/eval_dataset.pt")
    inp, lab = torch.load("dataset/mamba-ultrachat/train_dataset.pt").dataset.tensors
    # print(dataset)

    