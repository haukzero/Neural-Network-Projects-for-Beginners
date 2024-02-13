import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

####### CONST #######
BATCH_SIZE: int = 20
EPOCHS: int = 100
LEARNING_RATE: float = 0.0001
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
#####################

class Model:
    class CatVSDog(nn.Module):
        def __init__(self):
            super().__init__()
            # Conv layers and pooling layers
            self.conv_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                ),
                nn.Sequential(
                    nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                )
            ])
            # Fc layers
            self.fc_layers = nn.ModuleList([
                nn.Sequential(
                    # three 2x2 pooling layers: 200 / (2 ** 3) = 25
                    # 200x200 -> 25x25
                    # nodes: 128x25x25
                    nn.Linear(128 * 25 * 25, 512),
                    nn.Dropout(0.5),
                    nn.ReLU(),
                ),
                nn.Sequential(
                    nn.Linear(512, 256),
                    nn.Dropout(0.5),
                    nn.ReLU(),
                ),
                nn.Linear(256, 2),
            ])
            self.softmax = nn.Softmax(dim=1)
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for conv_layer in self.conv_layers:
                x = conv_layer(x)
            x = x.view(x.size(0), -1)
            for fc_layer in self.fc_layers:
                x = fc_layer(x)
            x = self.softmax(x)
            return x
    def __init__(self, load_pth: bool = False, pth_path: str | None = None):
        self.cat_vs_dog = Model.CatVSDog().to(DEVICE)
        self.optimizer = optim.Adam(self.cat_vs_dog.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.CrossEntropyLoss()
        if load_pth:
            self.cat_vs_dog.load_state_dict(torch.load(pth_path))
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.cat_vs_dog(x)
    def train(self, train_loader: DataLoader, save: bool = False, save_path: str | None = None):
        self.cat_vs_dog.train()
        for epoch in range(EPOCHS):
            for data in train_loader:
                features, labels = data[0].to(DEVICE), data[1].to(DEVICE)
                output: torch.Tensor = self.cat_vs_dog(features)
                loss: torch.Tensor = self.criterion(output, labels).to(DEVICE)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print(f"Epoch: {epoch}, Loss: {loss.item()}")
        if save:
            torch.save(self.cat_vs_dog.state_dict(), save_path)
    def test(self, test_loader: DataLoader) -> float:
        self.cat_vs_dog.eval()
        with torch.no_grad():
            correct: int = 0
            total: int = 0
            for data in test_loader:
                features, labels = data[0].to(DEVICE), data[1].to(DEVICE)
                output: torch.Tensor = self.cat_vs_dog(features)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = correct / total
            print(f"Accuracy: {accuracy}")
            return accuracy

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize(150),             # Scale the image to 150x150 pixels
    transforms.RandomVerticalFlip(),    # Flip image randomly
    transforms.RandomCrop(50),          # Crop the image to 50x50 pixels randomly
    transforms.RandomResizedCrop(200),  # Crop and scale the image to 200x200 pixels randomly
    transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),   # Color enhancement
    transforms.ToTensor(),              # Turn to Tensor
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])           # Normalize
])


if __name__ == '__main__':
    # Read
    root: str = 'imag'
    train_dataset = datasets.ImageFolder(root + '/train', transform=transform)
    test_dataset = datasets.ImageFolder(root + '/test', transform=transform)
    # Load
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    ### Get classes and classes_to_idxex
    # classes: list[str] = train_dataset.classes     # ['cat', 'dog']
    # classes_to_idxex: dict[str, int] = train_dataset.class_to_idx     # {'cat': 0, 'dog': 1}
    ### features: [20, 3, 150, 150]
    ### labels: [20]
    
    # pth_path: str = 'cat_vs_dog.pth'
    # model = Model(load_pth=True, pth_path=pth_path)
    # model.test(test_loader)
    model = Model()
    model.train(train_loader, save=True, save_path='cat_vs_dog.pth')
    model.test(test_loader)