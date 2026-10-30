import argparse
import os
import random as rand

from model import get_model, parse_prediction, DEVICE, TRANSFORM

import cv2
import torch


def get_random_imagepath(path = 'data/'):
    rand_country = rand.choice(os.listdir(os.path.join(path)))
    while not os.path.isdir(os.path.join(path, rand_country)):
        rand_country = rand.choice(os.listdir(os.path.join(path)))

    rand_image_name = rand.choice(os.listdir(os.path.join(path, rand_country, 'compressed_dataset', rand_country)))
    while not os.path.join(path, rand_country, 'compressed_dataset', rand_country, rand_image_name).endswith('.jpg'):
        rand_image_name = rand.choice(os.listdir(os.path.join(path, rand_country, 'compressed_dataset', rand_country)))
        
    return os.path.join(path, rand_country, 'compressed_dataset', rand_image_name)


def evaluate(model, n):
    datapath = 'data/'

    results = {
        'predictions': {
            'ghana': 0.0,
            'japan': 0.0,
            'malaysia': 0.0,
            'united_states': 0.0,
        },
        'ground_truth': {
            'ghana': 0.0,
            'japan': 0.0,
            'malaysia': 0.0,
            'united_states': 0.0,
        },
        'true_positives': {
            'ghana': 0.0,
            'japan': 0.0,
            'malaysia': 0.0,
            'united_states': 0.0,
        }
    }

    for _ in range(n):
        rand_country = rand.choice(os.listdir(os.path.join(datapath)))
        while not os.path.isdir(os.path.join(datapath, rand_country)):
            rand_country = rand.choice(os.listdir(os.path.join(datapath)))

        rand_image_name = rand.choice(os.listdir(os.path.join(datapath, rand_country, 'compressed_dataset', rand_country)))
        while not os.path.join(datapath, rand_country, 'compressed_dataset', rand_country, rand_image_name).endswith('.jpg'):
            rand_image_name = rand.choice(os.listdir(os.path.join(datapath, rand_country, 'compressed_dataset', rand_country)))

        results['ground_truth'][rand_country.lower().replace(' ', '_')] += 1
            
        image = cv2.imread(os.path.join(datapath, rand_country, 'compressed_dataset', rand_country, rand_image_name))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # type: ignore
        image = TRANSFORM(image)
        image = image.unsqueeze(0)
        image = image.to(DEVICE)

        with torch.no_grad():
            prediction = model(image)

        prediction = torch.softmax(prediction[0], dim=0).tolist()
        prediction = parse_prediction(prediction)

        country, _ = sorted(prediction.items(), key=lambda item: item[1], reverse=True)[0]
        results['predictions'][country] += 1

        if country == rand_country.lower().replace(' ', '_'):
            results['true_positives'][country] += 1

    print(f"{'Country':<16}{'Precision':>12}{'Recall':>10}{'F-score':>10}")
    print("-" * 48)

    for country in results['predictions']:
        tp = results['true_positives'][country]
        precision = tp / results['predictions'][country] if results['predictions'][country] else 0.0
        recall = tp / results['ground_truth'][country] if results['ground_truth'][country] else 0.0
        f_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        print(
            f"{country.replace('_', ' ').title():<16}"
            f"{precision:>12.4f}"
            f"{recall:>10.4f}"
            f"{f_score:>10.4f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str)
    parser.add_argument('--eval', type=int, default=1)
    args = parser.parse_args()

    model = get_model()
    model.to(DEVICE)

    weights = torch.load('weights.pt', map_location=DEVICE)
    model.load_state_dict(weights['model'])

    model.eval()

    if args.eval is not None:
        evaluate(model, args.eval)
        return

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