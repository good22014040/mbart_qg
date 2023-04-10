from torch.utils.data import DataLoader
import datasets
import json
import torch
from config import generate_max_length

class DRCD_dataset(torch.utils.data.Dataset):
  def __init__(self, encodings):
    self.encodings = encodings

  def __getitem__(self, idx):
    return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

  def __len__(self):
    return len(self.encodings.input_ids)

def read_drcd(path, num_limit = -1):
  with open(path, 'rb') as f:
    drcd_dict = json.load(f)

  contexts = []
  answers = []
  questions = []
  for group in drcd_dict['data']:
    for passage in group['paragraphs']:
      context = passage['context']
      for qa in passage['qas']:
        question = qa['question']
        for answer in qa['answers']:
          contexts.append(context)
          answers.append(answer['text'])
          questions.append(question)

          if num_limit != -1 and len(contexts) > num_limit:
            return contexts, answers, questions
  return contexts, answers, questions

def add_label(tokenizer, encodings, labels):
    lable_ids = []

    for label in labels:
        label_tokens = tokenizer(label + tokenizer.eos_token, max_length=generate_max_length, padding="max_length", truncation=True)
        lable_ids.append(label_tokens["input_ids"])
    encodings.update({'labels': lable_ids})

def get_dataset(path, tokenizer, num_limit = -1):
    contexts, answers, questions = read_drcd(path, num_limit = num_limit)
    encodings = tokenizer(
                  contexts, 
                  answers,
                  truncation=True, 
                  padding=True)
    add_label(tokenizer, encodings, questions)
    return DRCD_dataset(encodings)
