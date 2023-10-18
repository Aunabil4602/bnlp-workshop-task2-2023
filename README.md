# bnlp-task2

This repository contains the codes and external data used for the submission on task 2 of BLP Workshop @ EMNLP 2023.

Shared Task Link: https://github.com/blp-workshop/blp_task2

Workshop page Link: https://blp-workshop.github.io/

# external - datasets

- external_data_with_adjusted_labels.tsv - contains the several dataset(without banglabook) with adjusted labels. The datasets links are listed below.
  - [ "Emonoba: A dataset for analyzing fine-grained emotions on noisy bangla texts."](https://www.kaggle.com/datasets/saifsust/emonoba)
  - ["Cross-lingual sentiment classification in low-resource bengali language."](https://github.com/sazzadcsedu/BN-Dataset)
  - ["Bemoc: A corpus for identifying emotion in bengali texts."](https://github.com/avishek-018/BEmoC-Bengali-Emotion-Courpus)
  - ["Datasets for aspect-based sentiment analysis in
bangla and its baseline evaluation."](https://github.com/atik-05/Bangla_ABSA_Datasets)
  - https://www.kaggle.com/datasets/mahfuzahmed/banabsa
  - ["Abusive content detection
in transliterated bengali-english social media corpus."](https://github.com/sazzadcsedu/)
  - ["Emotion classification in a resource constrained language using
transformer-based approach."](https://github.com/omar-sharif03/NAACL-SRW-2021)
- external_data_banglabook_with_adjusted_label.tsv - contains the only banglabook dataset with adjusted labels.
  - ["Banglabook: A large-scale bangla dataset for sentiment analysis from book reviews."](https://github.com/mohsinulkabir14/banglabook)   
- paraphrasing_of_train_set_by_BanglaT5.tsv - contains the paraphrased data of the train set provided for the task by using [BanglaT5](https://arxiv.org/abs/2205.11081).

# codes

First, install the libraries with specific versions as mentioned in the requirements.txt file. 
Both files contain the CONFIG class with necessary hyper-parameter and other fields for training. Change the values accordingly and run the file.
