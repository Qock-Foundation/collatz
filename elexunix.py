import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange


n_letters = 7
delta = 1.0
device = 'cpu'


def read_subs(filename):
  return [line.strip().split(' -> ') for line in open(filename).readlines()]


def gen_samples(num_samples, length, subs):
  samples_in = torch.zeros((num_samples, length + 1), dtype=torch.long)
  samples_out = torch.zeros((num_samples, length + 1), dtype=torch.long)
  for it in range(num_samples):
    occ = []
    out1 = ''
    while not occ:
      letters = 'abefg'
      out1 = 'c' + ''.join(random.choice(letters) for _ in range(length - 2)) + 'd'
      for s1, s2 in subs:
        for i in range(length - len(s1) + 1):
          if out1[i:i + len(s1)] == s1:
            occ.append((i, s1, s2))
    ind, s1, s2 = random.choice(occ)
    out2 = out1[:ind] + s2 + out1[ind + len(s1):]
    samples_in[it, :len(out1)] = torch.tensor([ord(c) - ord('a') + 1 for c in out1])
    samples_out[it, :len(out2)] = torch.tensor([ord(c) - ord('a') + 1 for c in out2])
  return samples_in, samples_out


class Model(nn.Module):
  def __init__(self, hidden_size):
    super().__init__()
    self.rnn = nn.RNN(input_size=n_letters + 1, hidden_size=hidden_size, batch_first=True)
    self.post_rnn = nn.Sequential(
      nn.Linear(hidden_size, hidden_size),
      nn.ReLU(),
      nn.Linear(hidden_size, 1)
    )

  def forward(self, x):
    x = F.one_hot(x, num_classes=n_letters + 1).float()
    _, h = self.rnn(x)
    h = h.squeeze(0)
    return self.post_rnn(h).squeeze(-1)


if __name__ == '__main__':
  if len(sys.argv) != 2:
    print(f'usage: {sys.argv[0]} subs_filename')
    exit(1)
  model = Model(hidden_size=10).to(device)
  filename = 'models/recurrent_L50.pt'
  model.load_state_dict(torch.load(filename))
  test_size = 1000
  subs = read_subs(sys.argv[1])
  x, y = gen_samples(test_size, test_size, subs)
  x, y = x.to(device), y.to(device)
  p_x, p_y = model(x), model(y)
  for i in range(test_size):
    print(x[i])
    print(y[i])
    print(p_x[i].item(), p_y[i].item())
  accuracy = (p_y < p_x - delta).float().mean()
  mn, mx = torch.min(p_x), torch.max(p_x)
  print(f'accuracy: {accuracy.item():.2f}')
