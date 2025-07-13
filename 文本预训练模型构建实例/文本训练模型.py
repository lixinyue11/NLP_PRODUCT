import warnings

import torch

warnings.filterwarnings("ignore")
from transformers import AutoModelForMaskedLM,AutoTokenizer
model_checkpoint = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
# text = "This is a great [MASK]."
# inputs = tokenizer(text, return_tensors="pt")
# token_logits = model(**inputs).logits
# print(inputs["input_ids"])
# print(tokenizer)
# mask_token_index = torch.where(inputs["input_ids"] ==103)[1]
# mask_token_logits = token_logits[0, mask_token_index, :]
# # 对MASK所在位置找到他的TOP5预测结果
# top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
# for token in top_5_tokens:
#     print(f"'>>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")

'''--------'''
from datasets import load_dataset

imdb_dataset = load_dataset("imdb")
sample = imdb_dataset["train"].shuffle(seed=42).select(range(3))
# for row in sample:
#     print(f"\n'>>> Review: {row['text']}'")
#     print(f"'>>> Label: {row['label']}'")


'''映射'''
def tokenize_function(examples):#####获取ID
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result
tokenized_datasets = imdb_dataset.map(
    tokenize_function, batched=True, remove_columns=["text", "label"]#咱们是完形填空，不需要标签
)
'''长度'''
chunk_size = 128
'''文本切割'''
def group_texts(examples):
    # 拼接到一起
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # 计算长度
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // chunk_size) * chunk_size
    # 切分
    result = {
        k: [t[i: i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # 完型填空会用到标签的，也就是原文是啥
    result["labels"] = result["input_ids"].copy()
    return result


lm_datasets = tokenized_datasets.map(group_texts, batched=True)
from transformers import DataCollatorForLanguageModeling
'''长文本'''
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)#0.15是BERT人家说的，咱
# '''检查'''
# samples = [lm_datasets["train"][i] for i in range(2)]
# for sample in samples:
#     _ = sample.pop("word_ids")#不需要wordid
#     print(sample)
# for chunk in data_collator(samples)["input_ids"]:
#     print(f"\n'>>> {tokenizer.decode(chunk)}'")
#     print(len(chunk))
'''数据集进行采样，要不太慢了'''
train_size = 10000
test_size = int(0.1 * train_size)

downsampled_dataset = lm_datasets["train"].train_test_split(
    train_size=train_size, test_size=test_size, seed=42
)

'''训练 '''
from transformers import TrainingArguments

batch_size = 64
# 每一个epoch打印结果
logging_steps = len(downsampled_dataset["train"]) // batch_size
model_name = model_checkpoint.split("/")[-1]

training_args = TrainingArguments(
    output_dir=f"{model_name}-finetuned-imdb",#自己定名字
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    logging_steps=logging_steps,
    num_train_epochs=1,
    save_strategy='epoch',
)
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=downsampled_dataset["train"],
    eval_dataset=downsampled_dataset["test"],
    data_collator=data_collator,
)
import math
eval_results = trainer.evaluate()
trainer.train()


'''使用'''
from transformers import AutoModelForMaskedLM

model_checkpoint = "distilbert-base-uncased"
model = AutoModelForMaskedLM.from_pretrained("./distilbert-base-uncased-finetuned-imdb/checkpoint-157")
model = AutoModelForMaskedLM.from_pretrained("./distilbert-base-uncased-finetuned-imdb/checkpoint-157")
from transformers import AutoTokenizer
text='This is a great [MASK].'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

inputs = tokenizer(text, return_tensors="pt")
token_logits = model(**inputs).logits
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
mask_token_logits = token_logits[0, mask_token_index, :]
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

for token in top_5_tokens:
    print(f"'>>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")
