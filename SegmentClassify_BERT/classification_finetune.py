import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np

def preprocess(path_to_data_csv,model_name="distilbert/distilbert-base-uncased"):
    df = pd.read_csv(path_to_data_csv)
    df['label'] = df['label'].apply(lambda x: 1 if x == True else 0)
    dataset = Dataset.from_pandas(df)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)
    
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    split_dataset = tokenized_datasets.train_test_split(test_size=0.2)
    test_valid_split = split_dataset['test'].train_test_split(test_size=0.5)
    split_dataset['validation'] = test_valid_split['train']
    split_dataset['test'] = test_valid_split['test']
    return split_dataset



def train(data,model_name="distilbert/distilbert-base-uncased",save_path="models/finetuned_qualifications_classifier/"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    id2label = {0: "False", 1: "True"}
    label2id = {"False": 0, "True": 1}
    model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=2, id2label=id2label, label2id=label2id)
    # model = AutoModelForSequenceClassification.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    accuracy = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)
    
    training_args = TrainingArguments(
        output_dir=save_path,
        num_train_epochs=15,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs/',
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data['train'],
        eval_dataset=data['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

if __name__ == "__main__":
    classification_data_path = "../data/bert_finetune/classification_training_data.csv"
    preprocessed_data = preprocess(classification_data_path)
    train(preprocessed_data,save_path="models/finetuned_qualifications_classifier/")