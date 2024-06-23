import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Text(nn.Module):
    def __init__(self, args):
        super(CNN_Text, self).__init__()
        self.args = args

        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes

        if args.model_type == 'rand':
            self.embed = nn.Embedding(V, D)
        elif args.model_type == 'static':
            self.embed = nn.Embedding.from_pretrained(args.pretrained_embeddings, freeze=True)
        elif args.model_type == 'non_static':
            self.embed = nn.Embedding.from_pretrained(args.pretrained_embeddings, freeze=False)
        elif args.model_type == 'multichannel':
            self.embed_static = nn.Embedding.from_pretrained(args.pretrained_embeddings, freeze=True)
            self.embed_non_static = nn.Embedding.from_pretrained(args.pretrained_embeddings, freeze=False)

        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, C)

    def forward(self, x):
        if self.args.model_type == 'multichannel':
            x_static = self.embed_static(x)
            x_non_static = self.embed_non_static(x)
            x = torch.cat([x_static, x_non_static], dim=1)
        else:
            x = self.embed(x)

        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)

        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit
