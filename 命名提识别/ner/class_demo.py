import torch
import numpy as np


def get_token(input):
    # english = 'abcdefghijklmnopqrstuvwxyz0123456789'
    english = 'abcdefghijklmnopqrstuvwxyz'
    output = []
    buffer = ''
    for s in input:
        if s in english or s in english.upper():
            buffer += s
        else:
            if buffer: output.append(buffer)
            buffer = ''
            output.append(s)
    if buffer: output.append(buffer)
    return output


from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

model = AutoModelForTokenClassification.from_pretrained('D:/项目/咕泡项目/自然语言处理NLP/命名提识别/ner/checkpoint/model/class_bert-base-chinese-1000epoch')


from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')


if __name__ == '__main__':
    input_str = '今天天气很好'
    input_char = get_token(input_str)
    input_tensor = tokenizer(input_char, is_split_into_words=True, padding=True, truncation=True,
                             return_offsets_mapping=True, max_length=512, return_tensors="pt")
    input_tokens = input_tensor.tokens()
    offsets = input_tensor["offset_mapping"]
    ignore_mask = offsets[0, :, 1] == 0
    # print(input_tensor)
    input_tensor.pop("offset_mapping")  # 不剔除的话会报错
    outputs = model(**input_tensor)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()
    predictions = outputs.logits.argmax(dim=-1)[0].tolist()
    results = []
    print(predictions)
    tokens = input_tensor.tokens()
    idx = 0
    while idx < len(predictions):
        if ignore_mask[idx]:
            idx += 1
            continue
        pred = predictions[idx]
        # print(idx,pred,input_tokens[idx:idx+1])
        label = model.config.id2label[pred]
        a=input_tokens[idx:idx+1]
        print(label,a)
        if label != "O":
            # 不加B-或者I-
            label = label[2:]
            start = idx
            end = start + 1
            # 获取所有token I-label
            all_scores = []
            all_scores.append(probabilities[start][predictions[start]])
            print(idx,probabilities[start],predictions[start])
            while (
                    end < len(predictions)
                    and model.config.id2label[predictions[end]] == f"I-{label}"
            ):
                all_scores.append(probabilities[end][predictions[end]])
                end += 1
                idx += 1
                print( model.config.id2label[predictions[end]])
            # 得到是他们平均的
            score = np.mean(all_scores).item()
            word = input_tokens[start:end]
            results.append(
                {
                    "entity_group": label,
                    "score": score,
                    "word": word,
                    "start": start,
                    "end": end,
                }
            )
        idx += 1
    for i in range(len(results)):
        print(results[i])

    #print(results)
