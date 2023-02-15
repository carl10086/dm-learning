import math

from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from infra.train_toolkit import *

config = {
    'seed': 20230101,
    'batch_size': 10,
    'valid_ratio': 0.2,
    'n_epochs': 10,
    'learning_rate': 1e-2,
    'early_stop': 400,
    'feature_names': ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', \
                      'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
}


class HousePriceDataset(Dataset):
    def __init__(self, x, y=None) -> None:
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y)

    def __getitem__(self, index) -> T_co:
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


same_seed(config['seed'])

data = np.fromfile("/tmp/dataset/house/housing.data", sep=' ', dtype=np.float32)
feature_num = len(config['feature_names'])
data = data.reshape([data.shape[0] // feature_num, feature_num])
maximums, minimums = data.max(axis=0), data.min(axis=0)

# 记录数据的归一化参数，在预测时对数据做归一化
global max_values
global min_values
max_values = maximums
min_values = minimums

# 对数据进行归一化处理
for i in range(feature_num):
    data[:, i] = (data[:, i] - min_values[i]) / (maximums[i] - minimums[i])

print(f"""
data shape: {data.shape}
""")

train_data, valid_data = train_valid_split(data, config['valid_ratio'], config['seed'])

print(
    f"""
    shape of train_data: {train_data.shape},
    shape of valid_data: {valid_data.shape},
    """
)

y_train, y_valid = train_data[:, -1:], valid_data[:, -1:]
x_train, x_valid = train_data[:, :-1], valid_data[:, :-1]

print(
    f"""
    number of features: {x_train.shape[1]}
    """
)

feature_num = x_train.shape[1]

train_data_set = HousePriceDataset(x_train, y_train)
valid_data_set = HousePriceDataset(x_valid, y_valid)

train_loader = DataLoader(
    train_data_set,
    batch_size=config['batch_size'],
    shuffle=False,
    pin_memory=True
)

valid_loader = DataLoader(
    valid_data_set,
    batch_size=config['batch_size'],
    shuffle=False,
    pin_memory=True
)


class HousePriceModel(nn.Module):
    def __init__(self, input_dim) -> None:
        super(HousePriceModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            # nn.Sigmoid(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1)  # (B, 1) -> (B)
        return x


## Model
model = HousePriceModel(feature_num)

## Loss function
criterion = nn.MSELoss(reduction='mean')

optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'])

n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0

for epoch in range(n_epochs):
    model.train()
    loss_record = []

    # tqdm is a package to visualize your training progress.
    train_pbar = tqdm(train_loader, position=0, leave=True)

    device = 'cuda' if torch.cuda.is_available() else "cpu"
    for x, y in train_pbar:
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()  # Compute gradient(backpropagation).
        optimizer.step()  # Update parameters.
        step += 1
        loss_record.append(loss.detach().item())

        # Display current epoch number and loss on tqdm progress bar.
        train_pbar.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')
        train_pbar.set_postfix({'loss': loss.detach().item()})

    mean_train_loss = sum(loss_record) / len(loss_record)
    model.eval()  # Set your model to evaluation mode.
    loss_record = []
    for x, y in valid_loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred = model(x)
            loss = criterion(pred, y)

        loss_record.append(loss.item())

    mean_valid_loss = sum(loss_record) / len(loss_record)
    print(f'Epoch [{epoch + 1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')

    if mean_valid_loss < best_loss:
        best_loss = mean_valid_loss
        # torch.save(model.state_dict(), config['save_path'])  # Save your best model
        print('Saving model with loss {:.3f}...'.format(best_loss))
        early_stop_count = 0
    else:
        early_stop_count += 1

    if early_stop_count >= config['early_stop']:
        print('\nModel is not improving, so we halt the training session.')
