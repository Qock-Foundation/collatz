import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

class Dataset(nn.Module):
  def __init__(self, filename, split):
    self.l = open(filename).readlines()
    self.l, n = self.l[:-1], int(self.l[-1])
    assert len(self.l) == n
    if split == 'train':
      self.l = self.l[:9 * n // 10]
    elif split == 'test':
      self.l == self.l[9 * n // 10:]
    else:
      assert False
  def __getitem__(self, i):
    s, t = self.l[i].split()
    return torch.tensor(np.vectorize(ord)(np.array(list(s))) - ord('a'), dtype=torch.long), \
           torch.tensor(np.vectorize(ord)(np.array(list(t))) - ord('a'), dtype=torch.long)
  def __len__(self):
    return len(self.l)

train_dataset = Dataset('collatz_dataset_N1000000_L100', split='train')
test_dataset = Dataset('collatz_dataset_N1000000_L100', split='test')

def collate_fn(batch):
  L = max(max(t.shape[-1], s.shape[-1]) for t, s in batch)
  batch = (torch.stack([F.pad(t, (L - t.shape[-1], 0)) for t, s in batch]),
           torch.stack([F.pad(s, (L - s.shape[-1], 0)) for t, s in batch]))
  return batch

batch_size = 16
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, collate_fn=collate_fn, shuffle=True, pin_memory=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size, collate_fn=collate_fn, shuffle=True, pin_memory=True)

n_letters = 7

class Model(nn.Module):
  def __init__(self, n_channels, embedding_dim):
    super().__init__()
    self.embedder = nn.Embedding(num_embeddings=n_letters, embedding_dim=embedding_dim)
    self.stack = nn.Sequential(
      nn.Conv1d(in_channels=embedding_dim, out_channels=n_channels, kernel_size=3),
      nn.ReLU(),
      nn.Conv1d(in_channels=n_channels, out_channels=n_channels, kernel_size=5, stride=2),
      nn.ReLU(),
      nn.Conv1d(in_channels=n_channels, out_channels=1, kernel_size=5, stride=2),
    )
  def forward(self, x):
    x = self.embedder(x).transpose(-2, -1)
    x = self.stack(x)
    return x.mean(-1).squeeze(-1)

device = 'cuda'
H = Model(n_channels=512, embedding_dim=256).to(device)  # if it's energy, why don't we call it "H"?))
optimizer = torch.optim.Adam(H.parameters(), lr=1e-3)

losses = []
for s, t in tqdm(train_dataloader):
  s, t = s.to(device), t.to(device)
  E_s, E_t = H(s), H(t)
  loss = F.relu(E_t + 1 - E_s).mean()
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  losses.append(loss.item())
print('Loss:', np.mean(losses[-100:]))

plt.figure(figsize=(19.2, 10.8))
plt.plot(losses)
plt.xlabel('step')
plt.ylabel('loss')
plt.savefig('loss.png')
