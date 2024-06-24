import sys
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

class Dataset(nn.Module):
  def __init__(self, filename, split):
    self.l = open(filename).readlines()
    self.l, n = self.l[:-1], int(self.l[-1])
    assert len(self.l) == n
    n = 1_000_000
    if split == 'train':
      self.l = self.l[:9 * n // 10]
    elif split == 'test':
      self.l == self.l[9 * n // 10:]
    else:
      assert False
    self.l = sorted(self.l, key=len)
  def __getitem__(self, i):
    s, t = self.l[i].split()
    return torch.tensor(np.vectorize(ord)(np.array(list(s))) - ord('a') + 1, dtype=torch.long), \
           torch.tensor(np.vectorize(ord)(np.array(list(t))) - ord('a') + 1, dtype=torch.long)
  def __len__(self):
    return len(self.l)

filename = sys.argv[1]
train_dataset = Dataset(filename, split='train')
test_dataset = Dataset(filename, split='test')

def collate_fn(batch):
  L = max(max(t.shape[-1], s.shape[-1]) for t, s in batch)
  batch = (torch.stack([F.pad(t, (L - t.shape[-1], 0)) for t, s in batch]),
           torch.stack([F.pad(s, (L - s.shape[-1], 0)) for t, s in batch]))
  return batch

batch_size = 8
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, collate_fn=collate_fn, shuffle=True, pin_memory=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size, collate_fn=collate_fn, shuffle=True, pin_memory=True)

n_letters = 7 + 1  # blank

class Model(nn.Module):
  def __init__(self, num_layers, embedding_dim):
    super().__init__()
    self.num_layers = num_layers
    self.embedder = nn.Embedding(num_embeddings=n_letters, embedding_dim=embedding_dim)
    self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=embedding_dim, num_layers=num_layers, batch_first=True)
    self.post_rnn = nn.Sequential(
      nn.Linear(in_features=num_layers*embedding_dim, out_features=embedding_dim),
      nn.ReLU(),
      nn.Linear(in_features=embedding_dim, out_features=1),
    )
  def forward(self, x):
    x = self.embedder(x)
    B, L, H = x.shape
    x, h = self.rnn(x)
    h = h.transpose(0, 1).reshape(B, self.num_layers * H)
    return self.post_rnn(h)

device = 'cuda'
H = Model(num_layers=3, embedding_dim=2048).to(device)  # if it's energy, why don't we call it "H"?))
optimizer = torch.optim.Adam(H.parameters(), lr=1e-4)
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1-5/len(train_dataloader))  # e^5 ~= 150
scheduler = None  #torch.optim.lr_scheduler.LinearLR(optimizer, 1, 0, total_iters=len(train_dataloader))

losses = []
for s, t in tqdm(train_dataloader):
  s, t = s.to(device), t.to(device)
  E_s, E_t = H(s), H(t)
  if np.random.randint(10000) == 0:
    print(f'{E_s=}, {E_t=}')
  loss = F.relu(E_t + 1 - E_s).mean()
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  losses.append(loss.item())
  if scheduler is not None:
    scheduler.step()
losses = np.array(losses)
print('Loss:', losses[-100:].mean())

k = len(losses) // 100 * 100
losses = losses[:k].reshape(k // 100, 100).mean(-1)

plt.figure(figsize=(19.2, 10.8))
plt.plot(losses)
plt.xlabel('step, 100')
plt.ylabel('loss')
plt.savefig('loss.png')
