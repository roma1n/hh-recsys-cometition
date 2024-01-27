import polars as pl

from hh import (
    candidates,
    data,
    utils,
)


@utils.timeit
def get_history_baseline():
    return candidates.get_actions().join(
        candidates.get_likes(),
        on='user_id',
        how='outer',
    ).select(
        pl.exclude([
            'user_id_right',
            'user_id',
        ]),
        pl.coalesce([
            'user_id_right',
            'user_id',
        ]).alias('user_id'),
    ).join(
        candidates.get_applies(),
        on='user_id',
        how='outer',
    ).select(
        pl.exclude([
            'user_id_right',
            'user_id',
        ]),
        pl.coalesce([
            'user_id_right',
            'user_id',
        ]).alias('user_id'),
    ).select(
        pl.col('user_id'),
        pl.concat_list([
            pl.col('likes').fill_null([]),
            pl.col('actions').fill_null([]),
            pl.col('applies').fill_null([]),
        ]).list.unique(maintain_order=True).list.head(100).alias('predictions'),
    )


@utils.timeit
def get_history_plus_dssm():
    log = data.get_log().explode(
        'action_dt',
        'vacancy_id',
        'action_type',
    ).sort(
        'action_dt',
        descending=True,
    )
    likes = log.filter(
        pl.col('action_type') == 3,
    ).group_by(
        'user_id',
    ).agg(
        pl.col('vacancy_id').alias('likes'),
    )
    views = log.filter(
        pl.col('action_type') == 2,
    ).group_by(
        'user_id',
    ).agg(
        pl.col('vacancy_id').alias('views'),
    )
    applies = log.filter(
        pl.col('action_type') == 1,
    ).group_by(
        'user_id',
    ).agg(
        pl.col('vacancy_id').alias('applies'),
    )
    dssm = pl.read_parquet('data/dssm_prediction.pq').select(
        pl.col('user_id'),
        pl.col('dssm'),
    )
    needed = data.get_test_hh().select(
        pl.col('user_id')
    ).unique()
    res = needed.join(
        likes,
        on='user_id',
        how='left',
    ).join(
        applies,
        on='user_id',
        how='left',
    ).join(
        views,
        on='user_id',
        how='left',
    ).join(
        dssm,
        on='user_id',
        how='left',
    ).select(
        pl.col('user_id'),
        pl.col('likes').fill_null([]),
        pl.col('applies').fill_null([]),
        pl.col('views').fill_null([]),
        pl.col('dssm').fill_null([]),
    )
    return res.select(
        pl.col('user_id'),
        pl.concat_list([
            pl.col('likes').list.set_difference('applies'),
            pl.col('views').list.set_difference('applies'),
            pl.col('dssm').list.set_difference('applies'),
        ]).list.unique(maintain_order=True).list.head(100).alias('predictions'),
    )
