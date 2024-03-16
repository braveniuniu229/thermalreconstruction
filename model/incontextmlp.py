import torch
import torch.nn as nn
import torch.nn.functional as F

class Incontextmlp(nn.Module):
    def __init__(self, exp_num):
        super().__init__()
        self.in_channel = exp_num
        self.embedding_query = nn.Linear(16, 128)
        self.gelu1 = nn.GELU()
        self.embedding_samples = nn.Sequential(
            nn.Conv2d(self.in_channel, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(64 * 16 * 16, 256)

        self.l2 = nn.Linear(128, 256)
        self.l3 = nn.Linear(512, 2048)
        self.l4 = nn.Linear(2048, 4096)

    def forward(self, query, examples):
        q_emb = self.embedding_query(query)
        q_emb = self.gelu1(q_emb)
        q_emb = self.l2(q_emb)
        q_emb = self.gelu1(q_emb)  # Reuse the same GELU module for activation

        exp_emb = self.embedding_samples(examples)
        exp_emb = exp_emb.view(exp_emb.size(0), -1)  # Flatten the features from CNN
        exp_emb = self.fc(exp_emb)

        combined_emb = torch.cat((exp_emb, q_emb), dim=1)
        out = self.l3(combined_emb)
        out = self.gelu1(out)  # Reuse the same GELU module for activation
        out = self.l4(out)

        return out
if __name__ == "__main__":
    query = torch.randn(5,16)
    exp = torch.randn(5,4,64,64)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = Incontextmlp(4)
    query ,exp =query.to(device),exp.to(device)
    model.to(device)
    print(p.sum() for p in model.parameters())
    out = model(query,exp)
    out = out.view(out.size(0),-1)
    print(out.shape)