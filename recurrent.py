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
    n = 1000000
    if split == 'train':
      self.l = self.l[:9 * n // 10]  # just throw off 10% of data, why not
    else:
      assert False
    self.l = sorted(self.l, key=lambda s:len(s.split()[0]))
  def __getitem__(self, i):
    s, t = self.l[i].split()
    return torch.tensor(np.vectorize(ord)(np.array(list(s))) - ord('a') + 1, dtype=torch.long), \
           torch.tensor(np.vectorize(ord)(np.array(list(t))) - ord('a') + 1, dtype=torch.long)
  def __len__(self):
    return len(self.l)

filename = sys.argv[1]
train_dataset = Dataset(filename, split='train')
#test_dataset = Dataset(filename, split='test')

def collate_fn(batch):
  L = max(max(t.shape[-1], s.shape[-1]) for t, s in batch)
  batch = (torch.stack([F.pad(t, (L - t.shape[-1], 0)) for t, s in batch]),
           torch.stack([F.pad(s, (L - s.shape[-1], 0)) for t, s in batch]))
  return batch

batch_size = 128
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, collate_fn=collate_fn, shuffle=False, pin_memory=True)
#test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size, collate_fn=collate_fn, shuffle=False, pin_memory=True)

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
    h = h.view(B, self.num_layers * H)
    return self.post_rnn(h)

device = 'cpu'
H = Model(num_layers=1, embedding_dim=10).to(device)  # if it's energy, why don't we call it "H"?))
optimizer = torch.optim.Adam(H.parameters(), lr=1e-3)
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1-5/len(train_dataloader))  # e^5 ~= 150
scheduler = None  #torch.optim.lr_scheduler.LinearLR(optimizer, 1, 0, total_iters=len(train_dataloader))

losses_pot, losses_reg, losses_tot = [], [], []
for s, t in tqdm(train_dataloader):
  s, t = s.to(device), t.to(device)
  E_s, E_t = H(s), H(t)
  if np.random.randint(10000) == 0:
    print(f'{E_s=}, {E_t=}')
  loss_pot = F.relu(E_t + 1 - E_s).mean()
  loss_reg = F.relu(-E_t).mean() + F.relu(-E_s).mean()  #0.001 * ((E_t ** 2).mean() + (E_s ** 2).mean())
  loss_tot = loss_pot + loss_reg
  optimizer.zero_grad()
  loss_tot.backward()
  optimizer.step()
  losses_pot.append(loss_pot.item())
  losses_reg.append(loss_reg.item())
  losses_tot.append(loss_tot.item())
  if scheduler is not None:
    scheduler.step()
losses_pot = np.array(losses_pot)
losses_reg = np.array(losses_reg)
losses_tot = np.array(losses_tot)
print('Loss (potential part):', losses_pot[-1000:].mean())

k = len(losses_tot) // 10 * 10
losses_pot = losses_pot[:k].reshape(k // 10, 10).mean(-1)
losses_reg = losses_reg[:k].reshape(k // 10, 10).mean(-1)
losses_tot = losses_tot[:k].reshape(k // 10, 10).mean(-1)

plt.figure(figsize=(19.2, 10.8))
plt.plot(losses_pot, label='potential difference loss')
plt.plot(losses_reg, label='regularization loss')
plt.plot(losses_tot, label='model loss')
plt.legend()
plt.xlabel('step, 100')
plt.ylabel('loss')
plt.savefig('loss.png')
