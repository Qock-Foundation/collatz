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
    n = 100_000
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
  def __init__(self, embedding_dim):
    super().__init__()
    self.embedder = nn.Embedding(num_embeddings=n_letters, embedding_dim=embedding_dim)
    self.pos_encoding = nn.Parameter(torch.randn(200, embedding_dim))
#    self.stack = nn.Sequential(
#      nn.Conv1d(in_channels=embedding_dim, out_channels=n_channels, kernel_size=3),
#      nn.ReLU(),
#      nn.Conv1d(in_channels=n_channels, out_channels=n_channels, kernel_size=5, stride=2),
#      nn.ReLU(),
#      nn.Conv1d(in_channels=n_channels, out_channels=1, kernel_size=5, stride=2),
#    )
    self.stack = nn.Sequential(
      nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=1, dim_feedforward=4*embedding_dim, batch_first=True), num_layers=2),
      nn.Linear(in_features=embedding_dim, out_features=1),
    )
  def forward(self, x):
    x = self.embedder(x)#.transpose(-2, -1)
    #print(f'{x=}, {self.pos_encoding[-x.shape[-2]:]=}')
    x = x + self.pos_encoding[-x.shape[-2]:]
    x = self.stack(x)
    return x[:, :, 0].mean(-1)

device = 'cuda'
H = Model(embedding_dim=64).to(device)  # if it's energy, why don't we call it "H"?))
optimizer = torch.optim.Adam(H.parameters(), lr=1e-3)
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1-5/len(train_dataloader))  # e^5 ~= 150
scheduler = None  #torch.optim.lr_scheduler.LinearLR(optimizer, 1, 0, total_iters=len(train_dataloader))

losses = []
for s, t in tqdm(train_dataloader):
  s, t = s.to(device), t.to(device)
  E_s, E_t = H(s), H(t)
  if np.random.randint(1000) == 0:
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
