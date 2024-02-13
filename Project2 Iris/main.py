import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import torch
from torch import nn, optim

DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'

class Model:
    class BpNet(nn.Module):
        def __init__(self, n_feature: int, n_label: int, hidden: int | None = None):
            super(Model.BpNet, self).__init__()
            self.activate = nn.ReLU()
            self.softmax = nn.Softmax(dim=1)
            if hidden is None:
                self.layer = nn.Linear(n_feature, n_label)
            else:
                self.layer = nn.Sequential(
                    nn.Linear(n_feature, hidden),
                    self.activate,
                    nn.Linear(hidden, n_label),
                    self.activate,
                    self.softmax,
                )
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.layer(x)
            return x
    def __init__(self, n_feature: int, n_label: int, hidden: torch.Tensor | None = None, lr: float = 0.01, device: str = 'cpu'):
        self.net = Model.BpNet(n_feature, n_label, hidden).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
    def train(self, x: torch.Tensor, y: torch.Tensor, epoch: int = 1000) -> torch.Tensor:
        loss_arr = []
        for _ in range(epoch):
            yHat: torch.Tensor = self.net(x)
            loss: torch.Tensor = self.criterion(yHat, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_arr.append(loss.item())
        return torch.Tensor(loss_arr)
    def train_with_plot_loss(self, x: torch.Tensor, y: torch.Tensor, epoch: int = 1000,
                            show_frequency: int = 10, save: bool = False,
                            fig_name: str | None = None):
        px, py = [], []
        for i in range(epoch):
            yHat: torch.Tensor = self.net(x)
            loss: torch.Tensor = self.criterion(yHat, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            px.append(i)
            py.append(loss.item())
            if i % show_frequency == 0:
                plt.cla()
                plt.plot(px, py, 'r', lw=1)
                plt.xlabel('epoch')
                plt.ylabel('loss')
                plt.title('loss curve -- lastest: {}'.format(loss.item()))
                plt.show(block=False)
                plt.pause(0.01)
        if save:
            plt.savefig(fig_name)
    def test(self, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
        with torch.no_grad():
            yHat: torch.Tensor = self.net(x)
            loss: torch.Tensor = self.criterion(yHat, y)
            yHat = torch.max(yHat, 1)[1]
            acc = (yHat == y).sum().item() / y.size(0)
            print('loss: {}, acc: {}'.format(loss.item(), acc))
            return loss.item(), acc

# Read
iris = load_iris()
x: np.ndarray = iris['data']    # (150, 4)
y: np.ndarray = iris['target']  # (150,)
# feature names: sepal length (cm), sepal width (cm), petal length (cm), petal width (cm)
# target names: setosa, versicolor, virginica
feature_names = iris['feature_names']
target_names = iris['target_names']

# Select datasets randomly
indices = np.random.permutation(len(x))
train_indices = indices[:int(0.8 * len(indices))]
test_indices = indices[int(0.8 * len(indices)):]

train_x: np.ndarray = x[train_indices]
train_y: np.ndarray = y[train_indices]
test_x: np.ndarray = x[test_indices]
test_y: np.ndarray = y[test_indices]

# Turn to tensor
train_x = torch.FloatTensor(train_x).to(DEVICE)
train_y = torch.LongTensor(train_y).to(DEVICE)
test_x = torch.FloatTensor(test_x).to(DEVICE)
test_y = torch.LongTensor(test_y).to(DEVICE)

model = Model(4, 3, 5, device=DEVICE)
model.train(train_x, train_y)
model.test(test_x, test_y)