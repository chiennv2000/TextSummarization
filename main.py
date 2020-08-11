import os
import argparse
import time
import json
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import KFold
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertModel, BertTokenizer

from pyrouge import Rouge155
from preprocess import DatasetProcessor
from models import BertClassifier


parser = argparse.ArgumentParser(description='Text Summarization')
parser.add_argument('--dropout', type=float, default=0.4)
parser.add_argument('--kernel_num', type=int, default=100)
parser.add_argument('--kernel_sizes', type=str, default='1,2,3')
parser.add_argument('--pretrained_bert', type=str, default='bert-base-uncased')
parser.add_argument('--pretrained_sentence_bert', type=str, default='bert-base-nli-mean-tokens')
parser.add_argument('--rouge_path', type=str, default='/content/drive/My Drive/TextSumarization/pyrouge/tools/ROUGE-1.5.5/')
parser.add_argument('--lr', type=float, default=3e-5)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--max_sequence_length', type=int, default=128)
parser.add_argument('--n_folds', type=int, default=5)
parser.add_argument('--topk', type=int, default=4)
parser.add_argument('--train_path', type=str, default='./data/USAToday-CNN.json')
args = parser.parse_args()

r = Rouge155(args.rouge_path)

if torch.cuda.is_available():       
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def train(model, train_dataloader, epochs=args.epochs):
    print("Start training...\n")
    for epoch_i in range(epochs):
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Elapsed':^9}")
        print("-"*70)

        t0_batch = time.time()
        total_loss, batch_loss, batch_counts = 0, 0, 0

        model.train()
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            input_ids, attention_masks, label, comments_feature = tuple(t.to(device) for t in batch)
            model.zero_grad()

            logits = model(input_ids, attention_masks, comments_feature)

            loss = loss = loss_fn(logits, label)
            batch_loss += loss.item()
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                time_elapsed = time.time() - t0_batch
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {time_elapsed:^9.2f}")
        avg_train_loss = total_loss / len(train_dataloader)


def predict(model, test_id, processor, fold, evaluation):
    model.eval()
    with torch.no_grad():
        for idx, i in tqdm(enumerate(test_id, 0)):
            predicted_sentences = []
            paragraph = []
            all_comments_emb = processor.get_all_features_of_comments(i)
            for sent in processor.data[i]["document"]["sentences"]["sentence"]:
                text = tokenizer.cls_token + processor.data[i]["title"] + tokenizer.sep_token + sent["content"] + tokenizer.sep_token
                input_ids, attention_masks = processor.convert_example_to_feature(text)
                comment_feature = processor.get_feature_of_best_comment(sent["content"], all_comments_emb)
                logits = model(input_ids.to(device), attention_masks.to(device), comment_feature.to(device))
                score = F.softmax(logits, dim=1)
                predicted_sentences.append((sent["content"].lower(), score[0][1].item()))

        predicted_sentences = sorted(predicted_sentences, key=lambda x:x[1], reverse=True)
        if len(predicted_sentences) > args.topk:
            predicted_sentences = predicted_sentences[:args.topk]

        sumaries = []
        for sentence in predicted_sentences:
            sumaries.append(sentence[0])
        for sent in processor.data[i]["summary"]["sentences"]["sentence"]:
            text = sent["content"].lower()
            paragraph.append(text)

        with open("evaluation/system_summaries/" + str(fold) + "/text."+ str(idx) + ".txt", mode="w", encoding="utf-8") as fout_1:
                fout_1.write("\n".join(sumaries))

        with open("evaluation/model_summaries/"+ str(fold)+ "/text.A."+ str(idx) + ".txt", mode="w", encoding="utf-8") as fout_1:
                fout_1.write("\n".join(paragraph))

    r.system_dir = 'system_summaries/' + str(fold)
    r.model_dir = 'model_summaries/' + str(fold)
    r.system_filename_pattern = 'text.(\d+).txt'
    r.model_filename_pattern = 'text.[A-Z].#ID#.txt'
    output = r.convert_and_evaluate()
    evaluation.append(r.output_to_dict(output))
    print(output)

    return evaluation

def initialize_model(epochs=args.epochs):
    bert_classifier = BertClassifier(args)
    bert_classifier.to(device)
    optimizer = AdamW(bert_classifier.parameters(), lr=args.lr, eps=1e-8)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=50,
                                                num_training_steps=total_steps)
    loss_fn = nn.CrossEntropyLoss()
    return bert_classifier, optimizer, scheduler, loss_fn

def evaluate_model(evaluation):
    rouge_score = []
    for i in evaluation:
      rouge_score.append(list(i.values()))
    rouge_score = np.array(rouge_score)
    rouge_score = np.mean(rouge_score, axis=0)
    rouge_score = dict(zip(list(result[0].keys()), list(rouge_score)))
    print(json.dumps(rouge_score, indent=3))

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert)
    processor = DatasetProcessor(args, tokenizer)
    kf = KFold(n_splits=args.n_folds)
    i = 0
    evaluation = []
    for train_id, test_id in kf.split(processor.data):
        print("Training in fold :", i)
        train_dataloader = processor.load_training_data(train_id)
        bert_classifier, optimizer, scheduler, loss_fn = initialize_model()
        train(bert_classifier, train_dataloader)
        evaluation = predict(bert_classifier, test_id, processor, i, evaluation)
        i += 1
    evaluate_model(evaluation)