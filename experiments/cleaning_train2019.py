from pathlib import Path
import os

import pandas as pd
import tqdm


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'


def main():
    # input: 'input/train19_train19_search_top100_RANSAC.csv'
    # output: 'input/clean/train19_cleaned_verifythresh
    df_orig = load(verifth=0)
    for verifth in [20, 25, 30, 35, 40, 45, 50]:
        for freqth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            print(verifth, freqth)
            print('Load ransac results')
            df_ic = df_orig.copy()
            df_ic = df_ic[df_ic.cnt >= verifth]
            df_ic = df_ic.groupby('id', as_index=False).count()
            print('Load ransac results ... done')

            df = pd.read_csv(ROOT + 'input/train.csv',
                             usecols=['id', 'landmark_id'])
            df_ic = df_ic.merge(df, how='left', on='id')
            df_ic = df_ic[df_ic.cnt >= freqth]

            rows = []
            for landmark_id, df_part in tqdm.tqdm(
                    df_ic.groupby('landmark_id')):
                if len(df_part) < 2:
                    continue

                rows.append(dict(
                    landmark_id=landmark_id,
                    images=' '.join(df_part['id'].tolist())))

            Path(ROOT + 'input/clean').mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows)[['landmark_id', 'images']].to_csv(
                ROOT + 'input/clean/train19_cleaned_verifythresh' +
                f'{verifth}_freqthresh{freqth}.csv',
                index=False)


def load(verifth=20):
    df_ic = []
    for blockid in range(1, 33):
        df_part = pd.read_csv(
            ROOT + f'input/train19_train19_verified_blk{blockid}.csv',
            names=['id', 'result'],
            delimiter="\t")
        df_part.loc[:, 'cnt'] = df_part.result.apply(
            lambda x: int(x.split(':')[1]))
        # df_part = df_part[df_part.cnt >= verifth]
        df_ic.append(df_part)
    df_ic = pd.concat(df_ic, sort=False)
    return df_ic


if __name__ == '__main__':
    main()
