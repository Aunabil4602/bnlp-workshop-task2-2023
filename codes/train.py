# -*- coding: utf-8 -*-

####################### Installations

# !pip install -U accelerate
# !pip install -U transformers
# !pip install datasets
# !pip install git+https://github.com/csebuetnlp/normalizer.git
# !git clone https://github.com/blp-workshop/blp_task2.git

####################### Import

import random
import re
import string

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from datasets import load_metric
from normalizer import normalize
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import Trainer
from transformers import TrainingArguments


####################### Configs

class CONFIG:
    # Model settings
    model_name = 'csebuetnlp/banglabert_large'  # Best Model 1
    # model_name = 'amlan107/external-data-pretrained-banglabert-td20-dp-1-ep2' # Best Model 2 - set tokenizer_name to 'csebuetnlp/banglabert'
    # model_name = 'amlan107/external-data-pretrained-banglabert_large-td-20-dp-1-ep2' # Best Model 3
    tokenizer_name = 'csebuetnlp/banglabert_large'
    output_dir = 'finetuned-banglabert-v1'

    # Data File settings
    train_file = 'blp_task2/data/blp23_sentiment_train.tsv'
    validation_file = 'blp_task2/data/blp23_sentiment_dev.tsv'
    prediction_file = 'blp_task2/data/test/blp23_sentiment_test.tsv'

    # Training settings
    max_length = 128
    train_batch_size = 16
    val_batch_size = 16
    lr = 2e-5
    lr_scheduler_type = "linear"
    warmup_ratio = 0
    train_epochs = 3
    eval_strategy = "epoch"
    eval_steps = None
    weight_decay = 0.01
    save_limits = 20
    classifier_dropout = 0.1
    drop_token = 0.2

    # Other settings
    seed = 1234
    num_workers = 8


####################### SEED / Deterministic settings
def set_seed(seed=42, loader=None):
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


set_seed(seed=CONFIG.seed)

####################### Constants

LABEL2ID = {'Neutral': 0, 'Positive': 1, 'Negative': 2}
ID2LABEL = {LABEL2ID[k]: k for k in LABEL2ID.keys()}

####################### Processing data

train_df = pd.read_csv(CONFIG.train_file, sep='\t')
dev_df = pd.read_csv(CONFIG.validation_file, sep='\t')
test_df = pd.read_csv(CONFIG.prediction_file, sep='\t')

print(train_df.head())


def is_username(string):
    pattern = r'@([a-zA-Z0-9_]+)'
    matches = re.findall(pattern, string)
    return len(matches) > 0


PUNCTUATIONS = string.punctuation + '।'
URL = 'HTTPURL'
USER = 'USER'


def process(sentence):
    sentence = normalize(
        sentence,
        unicode_norm="NFKC",
        punct_replacement=None,
        url_replacement=URL,
        emoji_replacement=None,
        apply_unicode_norm_last=True
    )

    outputs = []
    for seg in sentence.split():
        if is_username(seg):
            seg = USER

        outputs.append(seg)

    return ' '.join(outputs)


processed_train_data = []
processed_train_label = []
for text, lab in zip(train_df['text'], train_df['label']):
    processed_train_data.append(process(text))
    processed_train_label.append(LABEL2ID[lab])

processed_val_data = []
processed_val_label = []
for text, lab in zip(dev_df['text'], dev_df['label']):
    processed_val_data.append(process(text))
    processed_val_label.append(LABEL2ID[lab])

processed_test_data = []
for text in test_df['text']:
    processed_test_data.append(process(text))

####################### Duplicates removal from Train, Val

unique_train_list = []
unique_train_label_list = []
print('original train: ' + str(len(processed_train_data)))

for sen, lab in zip(processed_train_data, processed_train_label):
    if sen not in unique_train_list:
        unique_train_list.append(sen)
        unique_train_label_list.append(lab)

print('unique train: ' + str(len(unique_train_list)))
processed_train_data, processed_train_label = unique_train_list, unique_train_label_list

unique_val_list = []
unique_val_label_list = []
print('original val: ' + str(len(processed_val_data)))

for sen, lab in zip(processed_val_data, processed_val_label):
    if sen not in unique_val_list:
        unique_val_list.append(sen)
        unique_val_label_list.append(lab)

print('unique val: ' + str(len(unique_val_list)))
processed_val_data, processed_val_label = unique_val_list, unique_val_label_list

####################### Tokenizer

tokenizer = AutoTokenizer.from_pretrained(CONFIG.tokenizer_name)

####################### Making Huggingface Dataset

train_dataset = Dataset.from_dict({'text': processed_train_data, 'label': processed_train_label})
val_dataset = Dataset.from_dict({'text': processed_val_data, 'label': processed_val_label})
test_dataset = Dataset.from_dict({'text': processed_test_data})

print(train_dataset)
print(val_dataset)
print(test_dataset)


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=CONFIG.max_length)


tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_val = val_dataset.map(preprocess_function, batched=True)
tokenized_test = test_dataset.map(preprocess_function, batched=True)


####################### Data Collator

class TokenDropDataCollatorWithPadding(DataCollatorWithPadding):
    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)
        self.tokenizer = tokenizer

    def __call__(self, features):
        updated_features = []
        for feature in features:
            updated_features.append(self.process(feature))

        batch = self.tokenizer.pad(
            updated_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch

    def process(self, feature):
        new_inputs_ids = [feature['input_ids'][0]]
        for token in feature['input_ids'][1:-1]:
            if random.random() >= CONFIG.drop_token:
                new_inputs_ids.append(token)
        new_inputs_ids.append(feature['input_ids'][-1])

        feature['input_ids'] = new_inputs_ids
        feature['token_type_ids'] = feature['token_type_ids'][:len(
            new_inputs_ids)]  # comment this while using xlm-roberta; it doesn't have this feature
        feature['attention_mask'] = feature['attention_mask'][:len(new_inputs_ids)]

        return feature


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
if CONFIG.drop_token > 0:
    data_collator = TokenDropDataCollatorWithPadding(tokenizer=tokenizer)

####################### Pretrained model

model = AutoModelForSequenceClassification.from_pretrained(CONFIG.model_name, num_labels=len(LABEL2ID),
                                                           classifier_dropout=CONFIG.classifier_dropout)
print(model)


####################### Metrics

def compute_metrics(eval_pred):
    load_accuracy = load_metric("accuracy")
    load_f1 = load_metric("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=predictions, references=labels, average='micro')["f1"]
    return {"accuracy": accuracy, "micro-f1": f1}


####################### Trainer

training_args = TrainingArguments(
    output_dir=CONFIG.output_dir,
    learning_rate=CONFIG.lr,
    lr_scheduler_type=CONFIG.lr_scheduler_type,
    warmup_ratio=CONFIG.warmup_ratio,
    per_device_train_batch_size=CONFIG.train_batch_size,
    per_device_eval_batch_size=CONFIG.val_batch_size,
    num_train_epochs=CONFIG.train_epochs,
    weight_decay=CONFIG.weight_decay,
    save_strategy=CONFIG.eval_strategy,
    evaluation_strategy=CONFIG.eval_strategy,
    seed=CONFIG.seed,
    fp16=True,
    dataloader_num_workers=CONFIG.num_workers,
    load_best_model_at_end=True,
    metric_for_best_model="micro-f1",
    greater_is_better=True,
    push_to_hub=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

####################### Inference & submission file generation

# trainer._load_from_checkpoint(CONFIG.output_dir + '/checkpoint-4156') # Load the best checkpoint
# trainer.data_collator = DataCollatorWithPadding(tokenizer=tokenizer) # load vanilla data collator
#
# predictions = trainer.predict(tokenized_test)
# pred_labels_ids = np.argmax(predictions.predictions, axis=-1)
# pred_labels = [ID2LABEL[id_] for id_ in pred_labels_ids]

# with open('task.tsv', 'w+') as f:
#   f.write('id\tlabel\n')
#   for id_,lab in zip(test_df['id'], pred_labels):
#     f.write(f'{id_}\t{lab}\n')

# !zip -r task.zip task.tsv
