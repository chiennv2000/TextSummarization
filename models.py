import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self, args):
        super(BertClassifier, self).__init__()
        D_in, D_out = 768, 2
        KERNEL_SIZES = [int(i) for i in args.kernel_sizes.split(",")]
        KERNEL_NUM = args.kernel_num
        DROPOUT = args.dropout
        self.bert = BertModel.from_pretrained(args.pretrained_bert, output_hidden_states=True)
        self.conv = nn.ModuleList([nn.Conv2d(1, KERNEL_NUM, (K, D_in)) for K in KERNEL_SIZES])
        self.dropout = nn.Dropout(DROPOUT)
        self.classifier = nn.Linear(len(KERNEL_SIZES)*KERNEL_NUM, D_out)


    def forward(self, input_ids, attention_mask, comments_feature):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        comments_feature = comments_feature.squeeze(1)
        cls_output = torch.cat((outputs[2][-1][:,0, ...].unsqueeze(1),outputs[2][-2][:,0, ...].unsqueeze(1), 
                               outputs[2][-3][:,0, ...].unsqueeze(1), outputs[2][-4][:,0, ...].unsqueeze(1), comments_feature.unsqueeze(1)), dim=1)    # (b, 5, 768)
        x = cls_output.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.conv]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits