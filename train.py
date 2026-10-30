from dataset import CountryDataset
from model import get_model, DEVICE, TRANSFORM

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


DATA_PATH = 'data/'
max_epochs = 100
batch_size = 32
validation_split = 0.3


def split_dataset(dataset):
    country_indices = {}
    for i, (label, _) in enumerate(dataset):
        country_indices.setdefault(int(label.argmax()), []).append(i)

    train_indices, val_indices = [], []
    for indices in country_indices.values():
        indices = [indices[i] for i in torch.randperm(len(indices)).tolist()]

        split = round(len(indices) * validation_split)
        train_indices += indices[split:]
        val_indices += indices[:split]

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def main():
    print(f'Using Device: {DEVICE}')

    model = get_model()
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(params=model.parameters())
    csl = torch.nn.CrossEntropyLoss()

    data = CountryDataset(path=DATA_PATH, transform=TRANSFORM)

    train_dataset, val_dataset = split_dataset(data)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2, shuffle=True, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2, shuffle=False, persistent_workers=True)
    
    for epoch in range(max_epochs):
        train_loss = 0
        val_loss = 0

        model.train()

        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

        for (y_true, x) in train_dataloader:
            y_true, x = y_true.to(DEVICE), x.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            y_pred = model(x)
            loss = csl(y_pred, y_true)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)

        model.eval()

        with torch.no_grad():
            for (y_true, x) in val_dataloader:
                y_true, x = y_true.to(DEVICE), x.to(DEVICE)

                y_pred = model(x)
                loss = csl(y_pred, y_true)

                val_loss += loss.item()

        val_loss /= len(val_dataloader)

        print(f'{epoch + 1}/{max_epochs}:\tTrain Loss: {train_loss}\tValidation Loss: {val_loss}')

    torch.save({
        'model': model.state_dict()
    }, 'weights.pt')


if __name__ == '__main__':
    main()