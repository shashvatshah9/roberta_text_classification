# Sequence classification using RoBerta

Using the Huggingface's pretrained [Roberta transformer](https://huggingface.co/docs/transformers/model_doc/roberta#transformers.RobertaForSequenceClassification), we train it on [ag_news](https://huggingface.co/datasets/ag_news) dataset.

For training this model, I went through a few iterations to optimize the best training and eval performance.
- Namely changing the learning rate policy, I trained it using COSINE scheduler with warmup period.
- Trained using dtype = pf16, to decrease the memory footprint of the model.
