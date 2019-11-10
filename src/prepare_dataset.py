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
    df.to_pickle(f'./{split}.pkl')


def generate_size_info_df(paths, df) -> pd.DataFrame:
    for path in tqdm(paths):
        id_ = str(path).split('/')[-1].replace('.jpg', '')
        img = cv2.imread(str(path))
        h, w, c = img.shape
        df.loc[id_, 'height'] = h
        df.loc[id_, 'width'] = w
    return df.reset_index().sort_values(by='id')


if __name__ == '__main__':
    csv2pkl(split='train')
    csv2pkl(split='train2018_r800')

    test19 = pd.read_csv('../input/test19/test.csv', index_col=[0]).sort_index()
    test_paths = list(Path('../input/test19/').glob('**/*.jpg'))
    test19 = generate_size_info_df(test_paths, test19)
    test19.to_pickle('../input/test19.pkl')

    index19 = pd.read_csv('../input/index19/index.csv', index_col=[0]).sort_index()
    index_paths = list(Path('../input/index19/').glob('**/*.jpg'))
    index19 = generate_size_info_df(index_paths, index19)
    index19.reset_index().sort_values(by='id').to_pickle('../input/index19.pkl')
