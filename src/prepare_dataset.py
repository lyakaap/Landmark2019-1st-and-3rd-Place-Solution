import pandas as pd
from pathlib import Path
import cv2
from tqdm import tqdm


def csv2pkl_train(split) -> None:
    df = pd.read_csv(f'../input/{split}.csv')
    paths = Path(f'../input/{split}/').glob('**/*.jpg')
    df_path = pd.DataFrame(paths, columns=['path'])
    df_path['path'] = df_path['path'].apply(lambda x: str(x.absolute()))
    df_path['id'] = df_path['path'].apply(
        lambda x: x.split('/')[-1].replace('.jpg', ''))
    df = df.merge(df_path, on='id')
    df = generate_size_info_df(paths, df)
    df.to_pickle(f'../input/{split}.pkl')


def generate_size_info_df(paths, df) -> pd.DataFrame:
    for path in tqdm(paths):
        id_ = str(path).split('/')[-1].replace('.jpg', '')
        img = cv2.imread(str(path))
        h, w, c = img.shape
        df.loc[id_, 'height'] = h
        df.loc[id_, 'width'] = w
    return df.reset_index().sort_values(by='id')


if __name__ == '__main__':
    csv2pkl(split='gld_v2/train')
    csv2pkl(split='gld_v1/train')

    test = pd.read_csv('../input/gld_v2/test.csv', index_col=[0]).sort_index()
    test_paths = list(Path('../input/gld_v2/test').glob('**/*.jpg'))
    test = generate_size_info_df(test_paths, test)
    test.to_pickle('../input/gld_v2/test.pkl')

    index = pd.read_csv('../input/gld_v2/index.csv', index_col=[0]).sort_index()
    index_paths = list(Path('../input/gld_v2/index/').glob('**/*.jpg'))
    index = generate_size_info_df(index_paths, index)
    index.reset_index().sort_values(by='id').to_pickle('../input/gld_v2/index.pkl')
