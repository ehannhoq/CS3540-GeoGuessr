from model import GeoGuessrModel
from dataset import CountryDataset


import torch
from torch.utils.data import DataLoader


DATA_PATH = 'data/'
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

max_epochs = 1000
batch_size = 32

def main():
    model = GeoGuessrModel().to(device=device)
    optimizer = torch.optim.AdamW(params=model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()


    data = CountryDataset(path=DATA_PATH)
    train_loader = DataLoader(dataset=data, batch_size=batch_size, num_workers=2, shuffle=True)

    model.train()

    for epoch in range(max_epochs):
        for (y_true, x) in train_loader:
            optimizer.zero_grad(set_to_none=True)

            y_pred = model(x.to(device))
            loss = loss_fn(y_pred, y_true.to(device))
            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    main()