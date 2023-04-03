# Sequence classification using RoBerta

Using the Huggingface's pretrained [Roberta transformer](https://huggingface.co/docs/transformers/model_doc/roberta#transformers.RobertaForSequenceClassification), we train it on [ag_news](https://huggingface.co/datasets/ag_news) dataset.

For training this model, I went through a few iterations to optimize the best training and eval performance.
- Namely changing the learning rate policy, I trained it using COSINE scheduler with warmup period.
- Trained using dtype = pf16, to decrease the memory footprint of the model.


The model used has pretained weights, except for some layer, where weights were not initialized from the model checkpoint at roberta-base and are newly initialized: ['classifier.out_proj.bias', 'classifier.dense.bias', 'classifier.out_proj.weight', 'classifier.dense.weight']. This is because the model is finetuned for the classification task.

I have used wandb for logging model metrics across different training runs.

<br>

![alt-text](https://github.com/shashvatshah9/roberta_text_classification/blob/main/roberta%20train%20metrics.png)
