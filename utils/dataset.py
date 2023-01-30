import gzip
import os
import pickle
import sys
from urllib.error import URLError
from urllib.request import urlretrieve

def show_progress(blk_num, blk_sz, tot_sz):
    """The reporthook function of "urlretrieve" function

    Args:
        blk_num (_type_): _description_
        blk_sz (_type_): _description_
        tot_sz (_type_): total size
    """
    percentage = 100. * blk_num * blk_sz / tot_sz
    print('Progress: %.1f %%' % percentage, end='\r', flush=True)

def download_url(url, file_path):
    """Download dataset from certain url

    Args:
        url (_type_): _description_
        file_path (_type_): _description_

    Raises:
        RuntimeError: _description_
    """
    # create directory if needed
    Dir = os.path.dirname(file_path)
    if not os.path.exists(Dir):
        os.makedirs(Dir)
    # download
    try:
        if os.path.exists(file_path):
            print("{} already exists.".format(file_path))
        else:
            print("Downloading {} to {}".format(url, file_path))
            try:
                urlretrieve(url, file_path, show_progress)
            except URLError:
                raise RuntimeError("Error downloading resource!")
            finally:
                print()
    except KeyboardInterrupt:
        print("Interrupted")


def prepare_dataset(data_dir: str, url: str):
    """Load dataset from certain url

    Args:
        data_dir (str): path to save data
        url (str): url of dataset

    Returns:
        _type_: _description_
    """
    save_path = os.path.join(data_dir, url.split("/")[-1])
    print("Preparing MNIST dataset ...")
    try:
        download_url(url, save_path)
    except Exception as e:
        print('Error downloading dataset: %s' % str(e))
        sys.exit(1)
    # load the dataset
    with gzip.open(save_path, "rb") as f:
        return pickle.load(f, encoding="latin1")
