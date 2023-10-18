# Installations

# !pip install -U accelerate
# !pip install -U transformers
# !pip install datasets huggingface_hub


# Imports

from transformers import PreTrainedModel, ElectraForMaskedLM, ElectraForPreTraining
from transformers import AutoTokenizer, BatchEncoding, ElectraConfig, DataCollatorForLanguageModeling
from urllib.parse import urlparse
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from datasets import load_metric
from transformers import TrainingArguments, Trainer
from torch import nn
from transformers import Trainer

import accelerate
import transformers
import numpy as np
import re
import string
import pandas as pd
import torch
import random
import torch.nn as nn

# Configs

# setup config class or, read config from args
class CONFIG:
  data_file = ['/content/pretraining_banglabook_balanced.txt', '/content/pretraining_no_banglabook.txt'] # train file, and will be splitted for validation
  validation_size = 10000

  # Model config
  train_dir = 'pretrained-banglabert-v3'
  generator_name = 'csebuetnlp/banglabert'
  discriminator_name = 'csebuetnlp/banglabert_generator'
  tie_embeddings = True
  generator_loss_weight = 1.0
  discriminator_loss_weight = 50.0

  # Training config
  max_length = 128
  train_batch_size = 64
  val_batch_size = 64
  lr = 1e-4
  train_epochs = 100
  eval_strategy = "epoch"
  eval_steps = None
  seed = 1234
  num_workers = 8
  weight_decay = 0.01
  warmup_ratio = 0.06
  save_limits = 20
  mlm_probability = 0.15

# SEED / Deterministic settings

def set_seed(seed = 42, loader = None):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    try:
        loader.sampler.generator.manual_seed(seed)
    except AttributeError:
        pass

set_seed(seed = CONFIG.seed)

# Tokenizer

tokenizer = AutoTokenizer.from_pretrained(CONFIG.discriminator_name)

# Creating Dataset + Tokenzing

# already normalized by csebuetnlp/normalizer
data = []
for file_name in CONFIG.data_file:
    for sen in open(file_name, 'r', encoding = 'utf-8').readlines():
        data.append(sen.strip())

random.shuffle(data)

split_index = CONFIG.validation_size # int(CONFIG.validation_size*(len(data)))
train_data = data[:-split_index]
val_data = data[-split_index:]

print(f'Train volumn: {len(train_data)}')
print(f'Validation volumn: {len(val_data)}')

train_dataset = Dataset.from_dict({'text': train_data})
val_dataset = Dataset.from_dict({'text': val_data})

def preprocess_function(examples):
  return tokenizer(examples["text"], truncation=True, max_length = CONFIG.max_length)

tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=['text'])
tokenized_val = val_dataset.map(preprocess_function, batched=True, remove_columns=['text'])

print(tokenized_train)

# Data collator

data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = True, mlm_probability = CONFIG.mlm_probability)

# Electra Combined Model

class ElectraForLanguageModelingModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.generator_model = ElectraForMaskedLM.from_pretrained(CONFIG.generator_name)
        self.discriminator_model = ElectraForPreTraining.from_pretrained(CONFIG.discriminator_name)

        self.generator_config = ElectraConfig.from_pretrained(CONFIG.generator_name)
        self.discriminator_config = ElectraConfig.from_pretrained(CONFIG.discriminator_name)

        # self.config = self.generator_config

        if CONFIG.tie_embeddings:
            self.tie_generator_and_discriminator_embeddings()

    def tie_generator_and_discriminator_embeddings(self):
        self.discriminator_model.set_input_embeddings(
            self.generator_model.get_input_embeddings()
        )

    def forward(self, input_ids, token_type_ids, attention_mask, labels):
        g_out = self.generator_model(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask, labels = labels)

        # creating inputs for discriminator by sampling(Multinomial)
        sample_probs = torch.softmax(g_out['logits'], dim=-1, dtype=torch.float32)
        sample_probs = sample_probs.view(-1, self.generator_config.vocab_size)
        sampled_tokens = torch.multinomial(sample_probs, 1).view(-1)
        sampled_tokens = sampled_tokens.view(g_out['logits'].shape[0], -1)
        # batch_sze may vary if its the last batch, so fetched the batch_size from outputs

        # setting inputsfor discriminator
        d_labels = labels.clone()
        d_input_ids = input_ids.clone()
        d_token_type_ids = token_type_ids.clone()
        d_attention_mask = attention_mask.clone()

        mask = d_labels.ne(-100)
        d_input_ids[mask] = sampled_tokens[mask]

        # setting labels for discriminator
        correct_preds = sampled_tokens == d_labels
        d_labels = mask.long()
        d_labels[correct_preds] = 0

        d_out = self.discriminator_model(input_ids = d_input_ids, token_type_ids = d_token_type_ids, attention_mask = d_attention_mask, labels = d_labels)

        # combined loss calculation
        combined_loss = CONFIG.generator_loss_weight * g_out['loss'] + CONFIG.discriminator_loss_weight * d_out['loss']

        return {
            "loss" : combined_loss,
            "logits" : d_out['logits']
        }

model = ElectraForLanguageModelingModel(CONFIG)

# Trainer

training_args = TrainingArguments(
   output_dir=CONFIG.train_dir,
   learning_rate=CONFIG.lr,
   per_device_train_batch_size=CONFIG.train_batch_size,
   per_device_eval_batch_size=CONFIG.val_batch_size,
   num_train_epochs=CONFIG.train_epochs,
   weight_decay=CONFIG.weight_decay,
   save_strategy=CONFIG.eval_strategy,
   evaluation_strategy=CONFIG.eval_strategy,
   seed=CONFIG.seed,
   fp16=True,
   dataloader_num_workers = CONFIG.num_workers,
   load_best_model_at_end=True,
   push_to_hub=False,
   remove_unused_columns = False
  #  full_determinism=True
)

trainer = Trainer(
   model=model,
   args=training_args,
   train_dataset=tokenized_train,
   eval_dataset=tokenized_val,
   tokenizer=tokenizer,
   data_collator=data_collator
)

trainer.train()