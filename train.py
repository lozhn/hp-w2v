import random
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

DEVICE = torch.device("mps")
import pickle

i2w = pickle.load(open("i2w.pkl", "rb"))
w2i = pickle.load(open("w2i.pkl", "rb"))
tokens = pickle.load(open("tokens.pkl", "rb"))

def window(tokens, n=5):
    for i in range(len(tokens) - n + 1):
        yield tokens[i:i+n]

class Net(nn.Module):
    def __init__(self, vocab_size, dim_size):
        super().__init__()
        # self.enc = nn.Embedding(vocab_size, dim_size, max_norm=1)  # doesnt work on MPS
        self.voc = vocab_size
        self.enc = nn.Linear(vocab_size, dim_size, bias=False)
        self.dec = nn.Linear(dim_size, vocab_size, bias=False)

    def forward(self, x):
        x = nn.functional.one_hot(x, self.voc)
        x = self.enc(x.float())
        x = self.dec(x)
        return x

    def encode(self, x):
        x = nn.functional.one_hot(x, self.voc)
        x = self.enc(x.float())
        return x

    def decode(self, x, n=10):
        x = self.dec(x)
        probas = torch.softmax(x, -1)
        return torch.flip(probas.argsort()[:, -n:], [-1])  # 0-th is the closest


def batch_cbow(win: list[str], dev=DEVICE):
    mid = len(win) // 2
    y = win[mid]
    x = win[:mid] + win[mid+1:]
    # print(x,y)
    y = torch.tensor([w2i[y]] * (len(win) - 1), device=dev)
    x = torch.tensor([w2i[w] for w in x], device=dev)
    return x, y

net = Net(vocab_size=len(i2w), dim_size=50).to(DEVICE)
loss_fn = torch.nn.NLLLoss()
optim = torch.optim.Adam(net.parameters())

def predict(model, dev=DEVICE):
    with torch.no_grad():
        logits = model(torch.tensor([w2i["harry"]], dtype=torch.long, device=dev))
    probas = torch.softmax(logits, -1).to("cpu")
    top = probas.argsort()[0]
    words = [i2w[i.item()] for i in top[-5:]]
    tqdm.write(f"pred for harry: {words}, {probas[0][top][-5:]}")

def epoch():
    predict(net)
    i = 0
    best = float("inf")
    best_pt = None
    for win in tqdm(window(tokens, 7)):
        x, y = batch_cbow(win)
        optim.zero_grad()
        pred = net(x)
        pred = torch.nn.functional.log_softmax(pred, -1)
        loss = loss_fn(pred, y)
        loss.backward()
        optim.step()
        if loss.item() < best:
            best = loss.item()
            best_pt = net.state_dict()
        if i % 5000 == 0:
            tqdm.write(f"loss: {loss.item()}")
            predict(net)
            torch.save(net.state_dict(), f"checkpoints/{i}.pt")
        i += 1
    # breakpoint()
    torch.save(best_pt, f"checkpoints/best_{best}.pt")
    net.load_state_dict(best_pt)
    print("BEST >>> ")
    predict(net)


if __name__ == "__main__":
    for i in range(10):
        print(f"epoch[{i+1:>2}]")
        epoch()
        predict(net)
