import torch
from train import Net
import pickle
DEVICE = torch.device("mps")

i2w = pickle.load(open("i2w.pkl", "rb"))
w2i = pickle.load(open("w2i.pkl", "rb"))
tokens = pickle.load(open("tokens.pkl", "rb"))

net = Net(vocab_size=len(i2w), dim_size=50).to(DEVICE)

net.load_state_dict(torch.load("checkpoints/best_2.1747548580169678.pt", weights_only=True))

inps = ["man"]
n = 5

def w2v(word):
    inp = torch.tensor([w2i[word]], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        embed = net.encode(inp)
    embed /= torch.linalg.vector_norm(embed)
    return embed

def v2w(vec, n=n):
    vec /= torch.linalg.vector_norm(vec)
    token = net.decode(vec, n=n)[0]
    return [i2w[t.item()] for t in token]


if __name__ == "__main__":
    import sys
    import re
    expr = sys.argv[1]
    args = re.split('(?<=\+|-)|(?=\+|-)', expr)
    acc = w2v(args.pop(0))
    while args:
        op = args.pop(0)
        arg = w2v(args.pop(0))
        if op == "+":
            acc += arg
        else:
            acc -= arg
    print( v2w(acc))
