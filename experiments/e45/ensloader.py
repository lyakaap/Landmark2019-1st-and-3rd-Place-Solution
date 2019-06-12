from easydict import EasyDict as edict

import ensemble_modelset_params as params
import datautil


def load_desc(name, weight_name, mode='retrieval'):
    # mode={retrieval, recognitino}
    modelset = getattr(params, name.upper())
    modelweight = getattr(params, weight_name.upper())

    if mode == 'retrieval':
        ids_index, feats_index = datautil.prepare_ids_and_feats(
            modelset['index'], modelweight, normalize=True)
        ids_test, feats_test = datautil.prepare_ids_and_feats(
            modelset['test'], modelweight, normalize=True)

        return edict({
            'ids_index': ids_index, 'ids_test': ids_test,
            'feats_index': feats_index, 'feats_test': feats_test,
        })

    elif mode == 'retrieval_dbatrain':
        ids_train, feats_train = datautil.prepare_ids_and_feats(
            modelset['train'], modelweight, normalize=True)

        return edict({
            'ids_train': ids_train, 'feats_train': feats_train,
        })

    elif mode == 'recognition':
        ids_train, feats_train = datautil.prepare_ids_and_feats(
            modelset['train'], modelweight, normalize=True)
        ids_test, feats_test = datautil.prepare_ids_and_feats(
            modelset['test'], modelweight, normalize=True)

        return edict({
            'ids_train': ids_train, 'ids_test': ids_test,
            'feats_train': feats_train, 'feats_test': feats_test,
        })

    else:
        raise RuntimeError(f"Invalid mode: {mode}")
