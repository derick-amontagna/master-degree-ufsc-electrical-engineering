from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
from common.utils import logger


# '..\\data\lgg-mri-segmentation\\kaggle_3m\\
def load_data(path):
    image_mri = []
    image_masks = glob(path + "\\*\\*_mask*")

    logger.info(f"Loading MRIs and Masks".center(70, "+"))
    for i in image_masks:
        image_mri.append(i.replace("_mask", ""))
    return image_mri, image_masks


def split_data(image_mri, image_masks, test_size=0.1, val_size=0.15):
    """Return train, test, val"""
    mri = pd.DataFrame(data={"mri": image_mri, "mask": image_masks})

    logger.info(
        f"Split into: train ({(1-test_size-val_size)*100}%), test({(test_size)*100}%), val({(val_size)*100}%)".center(
            70, "+"
        )
    )
    train, test = train_test_split(mri, test_size=test_size)
    train, val = train_test_split(train, test_size=val_size)

    return train, test, val


def show_mri_with_masks(image_mri, image_masks, rows: int = 3, columns: int = 3):

    fig = plt.figure(figsize=(10, 10))

    for i in range(1, rows * columns + 1):
        fig.add_subplot(rows, columns, i)
        img_path = image_mri[i]
        msk_path = image_masks[i]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        msk = cv2.imread(msk_path)
        plt.imshow(img)
        plt.imshow(msk, alpha=0.4, cmap="gray")
        plt.axis("off")

    plt.show()
