import numpy as np
import pandas as pd
from transformers import GPT2Tokenizer, GPT4ForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

print("Carregando dados...")
df = pd.read_csv('iot23_combined_new.csv')

print("Limpando dados...")
df['orig_bytes'] = df['orig_bytes'].replace('-', np.nan)
df['resp_bytes'] = df['resp_bytes'].replace('-', np.nan)
df['orig_bytes'] = df['orig_bytes'].astype('float64')
df['resp_bytes'] = df['resp_bytes'].astype('float64')
df['duration'] = df['duration'].astype(str)
df['label_numeric'] = df['label'].astype('category').cat.codes

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
pad_token = tokenizer.eos_token
tokenizer.pad_token = pad_token

def create_sentence(row):
    return f"{row['id.orig_h']} sent {row['orig_bytes']} bytes to {row['id.resp_h']} over {row['proto']} protocol"
df['sentence'] = df.apply(create_sentence, axis=1)

train_df, test_df = train_test_split(df, test_size=0.1)
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

def tokenize_and_add_labels(examples):
    tokenized_inputs = tokenizer(
        examples['sentence'],
        padding="longest",
        truncation=True,
        max_length=512
    )
    tokenized_inputs['labels'] = examples['label_numeric']
    return tokenized_inputs

train_dataset = train_dataset.map(tokenize_and_add_labels, batched=True)
test_dataset = test_dataset.map(tokenize_and_add_labels, batched=True)
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

num_labels = df['label_numeric'].nunique()
model = GPT4ForSequenceClassification.from_pretrained('gpt4', num_labels=num_labels)
model.config.pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)

training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=lambda p: {'accuracy': accuracy_score(p.label_ids, p.predictions.argmax(-1))}
)

trainer.train()
print("Treinamento concluído!")
