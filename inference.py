import argparse
import os
import random as rand

from model import get_model, parse_prediction, DEVICE, TRANSFORM

import cv2
import torch


def get_random_imagepath(path = 'data/'):
    countries = []
    for country in os.listdir(path):
        if not os.path.isdir(os.path.join(path, country)):
            continue

        countries.append(country)

    country = rand.choice(countries)

    image_names = []

    for image_name in os.listdir(os.path.join(path, country)):
        if not os.path.join(path, country, image_name).endswith('.jpg'):
            continue

        image_names.append(image_name)

    return os.path.join(path, country, rand.choice(image_names))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str)
    args = parser.parse_args()

    model = get_model()
    model.to(DEVICE)

    weights = torch.load('weights.pt', map_location=DEVICE)
    model.load_state_dict(weights['model'])

    model.eval()

    if args.image is not None:
        image = cv2.imread(args.image)
    else:
        imagepath = get_random_imagepath()
        print(f'No image specified, using random image: {imagepath}')
        image = cv2.imread(imagepath)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # type: ignore
    image = TRANSFORM(image)
    image = image.unsqueeze(0)
    image = image.to(DEVICE)

    with torch.no_grad():
        prediction = model(image)

    prediction = torch.softmax(prediction[0], dim=0).tolist()
    prediction = parse_prediction(prediction)

    for country, confidence in sorted(prediction.items(), key=lambda item: item[1], reverse=True):
        print(f'{country.replace('_', ' ').title()}: {confidence * 100 :.2f}%')


if __name__ == '__main__':
    main()