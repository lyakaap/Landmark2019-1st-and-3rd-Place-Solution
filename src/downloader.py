# Fast image downloader using this trick:
# https://www.kaggle.com/c/landmark-recognition-challenge/discussion/49703
# And you can change target size that you prefer.

# Reference:
# https://www.kaggle.com/c/landmark-recognition-challenge/discussion/48895
# For 256,256 this should be 22 GB
# For 224,224 this should be 16.8 GB
# For 139,139 this should be 6.5 GB
# For 128,128 this should be 5.5 GB
# For 96,96 this should be 3.1 GB
# For 64,64 this should be 1.4 GB

import multiprocessing
import os
from io import BytesIO
from urllib import request
import pandas as pd
import re
import tqdm
from PIL import Image
import click

CSV_PATH, OUT_DIR = '../input/gld_v1/train.csv', '../input/gld_v1/train'  # recognition challenge
# CSV_PATH, OUT_DIR = '../input/index.csv', '../input/index'  # retrieval challenge
# CSV_PATH, OUT_DIR = '../input/test.csv', '../input/test'  # test data

TARGET_SIZE = 800  # image resolution to be stored
MIN_SIZE_SHORTER = 256
IMG_QUALITY = 90  # JPG quality
NUM_WORKERS = multiprocessing.cpu_count()  # Num of CPUs


def overwrite_urls(df):
    def reso_overwrite(url_tail):
        pattern = 's[0-9]+'
        search_result = re.match(pattern, url_tail)
        if search_result is None:
            return url_tail
        else:
            return 's{}'.format(TARGET_SIZE)

    def join_url(parsed_url, s_reso):
        parsed_url[-2] = s_reso
        return '/'.join(parsed_url)

    parsed_url = df.url.apply(lambda x: x.split('/'))
    resos = parsed_url.apply(lambda x: reso_overwrite(x[-2]))

    overwritten_df = pd.concat([parsed_url, resos], axis=1)
    overwritten_df.columns = ['url', 's_reso']
    df['url'] = overwritten_df.apply(lambda x: join_url(x['url'], x['s_reso']), axis=1)
    return df


def parse_data(df):
    key_url_list = [line[:2] for line in df.values]
    return key_url_list


def download_image(key_url):
    (key, url) = key_url
    filename = os.path.join(OUT_DIR, '{}.jpg'.format(key))

    if os.path.exists(filename):
        # print('Image {} already exists. Skipping download.'.format(filename))
        return 0

    try:
        response = request.urlopen(url)
        image_data = response.read()
    except:
        print('Warning: Could not download image {} from {}'.format(key, url))
        return 1

    try:
        pil_image = Image.open(BytesIO(image_data))
    except:
        print('Warning: Failed to parse image {}'.format(key))
        return 1

    try:
        pil_image = pil_image.convert('RGB')
    except:
        print('Warning: Failed to convert image {} to RGB'.format(key))
        return 1

    try:
        shorter_side = min(pil_image.width, pil_image.height)
        longer_side = max(pil_image.width, pil_image.height)

        if shorter_side < MIN_SIZE_SHORTER:
            print('Warning: Image size is too small {}, size: {}x{}'.format(key, pil_image.width, pil_image.height))
            return 0

        if longer_side > TARGET_SIZE:
            factor = TARGET_SIZE / longer_side

            if round(shorter_side * factor) < MIN_SIZE_SHORTER:
                print('Warning: Image size is too small {}, size: {}x{}'.format(key, pil_image.width, pil_image.height))
                return 0

            tw, th = round(pil_image.width * factor), round(pil_image.height * factor)
            pil_image = pil_image.resize((tw, th))
            print('size: {}x{}'.format(key, pil_image.width, pil_image.height))
            # return 0
    except:
        print('Warning: Failed to resize image {}'.format(key))
        return 1

    try:
        pil_image.save(filename, format='JPEG', quality=IMG_QUALITY)
    except:
        print('Warning: Failed to save image {}'.format(filename))
        return 1

    return 0


def loader(df):
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)

    key_url_list = parse_data(df)
    pool = multiprocessing.Pool(processes=NUM_WORKERS)
    failures = sum(tqdm.tqdm(pool.imap_unordered(download_image, key_url_list),
                             total=len(key_url_list), ncols=70))
    print('Total number of download failures:', failures)
    pool.close()
    pool.terminate()


@click.group()
def cli():
    pass


@cli.command()
def download():
    df = pd.read_csv(CSV_PATH).query('url != "None"')
    loader(overwrite_urls(df))


if __name__ == '__main__':
    cli()
