import hnswlib
import lightfm
import numpy as np
import pickle
import polars as pl
from scipy import sparse

from hh import (
    data,
    utils,
)

EMBED_DIM = 256
N_USERS = 1177422
N_ITEMS = 2734129
N_EPOCH = 32
MODEL_FILE = 'data/lightfm_training={}.pickle'

def train(training=False, sampling=False):
    log = data.get_log(training=training).select(
        'user_id',
        'vacancy_id',
    ).explode(
        'vacancy_id',
    ).select(
        pl.col('user_id').str.slice(2).cast(pl.Int64).alias('uid'),
        pl.col('vacancy_id').str.slice(2).cast(pl.Int64).alias('iid'),
    )

    sample = log
    if sampling:
        sample = log.filter(
            (pl.col('uid').hash().mod(10).eq(1)) & (pl.col('iid').hash().mod(10).eq(1))
        )
    print('Train sample len:', len(sample))

    train = sparse.coo_matrix(
        (
            np.ones(len(sample)),
            (
                sample['uid'].to_list(),
                sample['iid'].to_list(),
            ),
        ),
        shape=(N_USERS, N_ITEMS),
    )
    model = lightfm.LightFM(
        loss='warp',
        no_components=EMBED_DIM,
    )
    model.fit(
        train,
        epochs=N_EPOCH,
        verbose=2,
    )
    with open(MODEL_FILE.format(training), 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)


def build_index(vacancy_embeddings):
    print('Building index')
    p = hnswlib.Index(
        space='cosine',
        dim=EMBED_DIM,
    )
    p.init_index(max_elements=N_ITEMS, ef_construction=1000, M=64)
    p.add_items(vacancy_embeddings, np.arange(N_ITEMS))
    return p


def get_predictions_by_index(p, user_embeddings):
    print('Executing KNN query')
    labels, distances = p.knn_query(user_embeddings, k=300)
    return pl.DataFrame().with_columns(
        user_id=np.arange(N_USERS),
        fm=labels,
        fm_distances=distances,
    ).explode(
        'fm',
        'fm_distances',
    ).select(
        pl.concat_str(pl.lit('v_'), pl.col('fm').cast(pl.String)).alias('fm'),
        pl.col('fm_distances'),
        pl.concat_str(pl.lit('u_'), pl.col('user_id').cast(pl.String)).alias('user_id'),
    ).group_by(
        'user_id',
    ).agg(
        pl.col('fm'),
        pl.col('fm_distances'),
    )


@utils.timeit
def get_predictions(training=False):
    print('Loading model')
    with open(MODEL_FILE.format(training), 'rb') as f:
        model = pickle.load(f)

    return get_predictions_by_index(
        p=build_index(vacancy_embeddings=model.item_embeddings),
        user_embeddings=model.user_embeddings,
    )
