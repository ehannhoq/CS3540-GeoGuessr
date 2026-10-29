import os

import cv2
import numpy as np
from torch.utils.data import Dataset

class CountryDataset(Dataset):
    def __init__(self, path, transform = None):
        self.data = []
        self.transform = transform

        for country in os.listdir(path):
            if not os.path.isdir(os.path.join(path, country)):
                continue

            for image_name in os.listdir(f'data/{country}'):
                if not os.path.join(path, country, image_name).endswith('.jpg'):
                    continue
                
                self.data.append((self.__country_to_arr__(country), os.path.join(path, country, image_name)))
                

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        country, image_path = self.data[index]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # type: ignore

        if self.transform:
            image = self.transform(image)

        return country.astype(np.float32), image


    def __country_to_arr__(self, country):
        if country == 'Ghana':
            return np.array([1.0, 0.0, 0.0, 0.0])

        if country == 'Japan':
            return np.array([0.0, 1.0, 0.0, 0.0])

        if country == 'Malaysia':
            return np.array([0.0, 0.0, 1.0, 0.0])

        if country == 'United States':
            return np.array([0.0, 0.0, 0.0, 1.0])


if __name__ == '__main__':
    dataset = CountryDataset('data/')

