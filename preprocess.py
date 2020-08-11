import json
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sentence_transformers import SentenceTransformer

class DatasetProcessor(object):

    def __init__(self, args, tokenizer):
        self.tokenizer = tokenizer
        self.MAX_LEN = args.max_sequence_length
        self.batch_size = args.batch_size
        self.data = self.load_from_path(args.train_path)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.sentence_bert = SentenceTransformer(args.pretrained_sentence_bert)
        for param in self.sentence_bert.parameters():
            param.requires_grad = False

    def load_from_path(self, dataset_path):
        with open(dataset_path, mode="r", encoding="utf-8") as f:
            data = json.loads(f.read())["posts"]["post"]
        return data

    def get_all_features_of_comments(self, i):
        comments = []
        for cmt in self.data[i]["comments"]["comment"]["sentences"]["sentence"]:
            if "-in_summary" in cmt:
                if cmt["-in_summary"] == "YES":
                    comments.append(cmt["content"])
        comments_emb = self.sentence_bert.encode(comments)
        comments_emb = torch.tensor(comments_emb, dtype=torch.float)
        return comments_emb
      
    def get_feature_of_best_comment(self, sentence, comments_emb):
        sentence_embd = self.sentence_bert.encode([sentence])
        sentence_embd = torch.tensor(sentence_embd, dtype=torch.float)
        measure_similarity = self.cos(sentence_embd, comments_emb)
        cmt_index = torch.topk(measure_similarity, k=1)[1]
        return comments_emb[cmt_index]
    
    def load_training_data(self, idx_training_data):
        input_ids = []
        attention_masks = []
        label = []
        comments_feature = []
        for i in tqdm(idx_training_data):
            all_comments_emb = self.get_all_features_of_comments(i)
            for sent in self.data[i]["document"]["sentences"]["sentence"]:
                if "-in_summary" in sent:
                    encoded_sent = self.tokenizer.encode_plus(
                                            text = self.tokenizer.cls_token + self.data[i]["title"] + self.tokenizer.sep_token + sent["content"] + self.tokenizer.sep_token,
                                            add_special_tokens=False,
                                            max_length=self.MAX_LEN,
                                            pad_to_max_length=True,        
                                            return_attention_mask=True)

                    input_ids.append(encoded_sent.get('input_ids'))
                    attention_masks.append(encoded_sent.get('attention_mask'))
                    comments_feature.append(self.get_feature_of_best_comment(sent["content"], all_comments_emb))
                    if sent["-in_summary"] == "YES":
                        label.append(1)
                    else:
                        label.append(0)
                    
        comments_feature = torch.stack(comments_feature, dim=0)
        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)
        label = torch.tensor(label)

        return self.create_data_loader(input_ids, attention_masks, label, comments_feature)

    def create_data_loader(self, input_ids, attention_masks, label, comments_feature):
        train_data = TensorDataset(input_ids, attention_masks, label, comments_feature)
        train_dataloader = DataLoader(train_data, batch_size=self.batch_size)
        return train_dataloader

    def convert_example_to_feature(self, text):
        encoded_sent = self.tokenizer.encode_plus(
                    text=text,
                    add_special_tokens=False,
                    max_length=self.MAX_LEN,
                    pad_to_max_length=True,        
                    return_attention_mask=True)
        input_ids = encoded_sent.get('input_ids')
        attention_masks = encoded_sent.get('attention_mask')
        return torch.tensor(input_ids, dtype=torch.long).unsqueeze(0), torch.tensor(attention_masks, dtype=torch.long).unsqueeze(0)