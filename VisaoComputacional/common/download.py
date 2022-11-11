import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
from utils import logger


api = KaggleApi()
api.authenticate()


# Signature: dataset_download_files(dataset, path=None, force=False, quiet=True, unzip=False)
logger.info(f"Executing download data".center(70, "+"))
api.dataset_download_files("mateuszbuda/lgg-mri-segmentation", path=".\\data\\", quiet=False, unzip=True)

logger.info(f"Deleting .zip".center(70, "+"))
file_list = os.listdir(".\\data")
file_zip = [x for x in file_list if x.endswith(".zip")]

for zip in file_zip:
    os.remove(f".\\data\\{zip}")
