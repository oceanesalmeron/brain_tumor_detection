"""
Module implemented to build and pre-process a dataset containing images.
"""

import os

import cv2
import pandas as pd

ROOT = './data'

def build_dataset() -> pd.DataFrame:
    """
    Build a dataset from mri images.

    Returns
    -------
    A Pandas dataframe containing the data.
    """
    labels = []
    paths = []

    for path, _, files in os.walk(ROOT):
        for name in files:
            labels.append(os.path.basename(path))
            paths.append(os.path.join(path, name))

    # df = pd.DataFrame({'paths': paths, 'labels': labels})
    # return df.iloc[1: , :]
    paths.pop(0)
    labels.pop(0)
    return paths, labels


def process_images(img_paths: pd.Series) -> pd.Series:
    """
    Pre process images.

    Returns
    -------
    A Pandas Series of processed images.
    """
    images = []

    for path in img_paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224,224))
        images.append(img)

    return images


if __name__ == '__main__':
    data = build_dataset()
    data['data'] = process_images(data['paths'])
    data['labels'], labels = data['labels'].factorize()
    print(data.head(-10))
