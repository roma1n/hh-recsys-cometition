import polars as pl

from hh import (
    data,
    utils,
)


@utils.timeit
def get_reversed_events_log():
    return data.get_log().select(
        pl.col('user_id'),
        pl.col('vacancy_id'),
        pl.col('action_type'),
        pl.col('action_dt'),
    ).explode(
        pl.col('vacancy_id'),
        pl.col('action_type'),
        pl.col('action_dt'),
    ).sort(
        pl.col('action_type'),
        descending=True,
    )


@utils.timeit
def get_applies_flatten():
    return get_reversed_events_log().filter(
        pl.col('action_type') == 1,
    ).select(
        pl.col('user_id'),
        pl.col('vacancy_id'),
        pl.col('action_type').alias('apply_action'),
    )


@utils.timeit
def get_applies():
    return get_applies_flatten().select(
        pl.col('user_id'),
        pl.col('vacancy_id'),
    ).group_by(
        'user_id',
    ).agg(
        pl.col('vacancy_id').alias('applies')
    )


@utils.timeit
def get_actions():
    return get_reversed_events_log().join(
        get_applies_flatten(),
        on=['user_id', 'vacancy_id'],
        how='left',
    ).filter(
        pl.col('apply_action').is_null(),
    ).group_by(
        pl.col('user_id'),
    ).agg(
        pl.col('vacancy_id').alias('actions'),
    )


@utils.timeit
def get_likes():
    return get_reversed_events_log().filter(
        pl.col('action_type') == 3,
    ).join(
        get_applies_flatten(),
        on=['user_id', 'vacancy_id'],
        how='left',
    ).filter(
        pl.col('apply_action').is_null(),
    ).group_by(
        pl.col('user_id'),
    ).agg(
        pl.col('vacancy_id').alias('likes'),
    )


@utils.timeit
def get_dssm():
    return pl.read_parquet(
        'data/dssm_prediction.pq'
    )


@utils.timeit
def get_application_candidates(training=False):
    log = data.get_log()
    if training:
        targets = pl.read_parquet('data/dssm_train.pq').select(
            pl.col('session_id'),
        ).unique()
        log = log.join(
            targets,
            on='session_id',
            how='anti',
        )
    log = log.explode(
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
    dssm = get_dssm().select(
        pl.col('user_id'),
        pl.col('dssm'),
        pl.col('dssm_distances'),
    )
    needed = data.get_log() if training else data.get_test_hh()
    needed = needed.select(
        pl.col('user_id')
    ).unique()
    return needed.join(
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
        pl.col('dssm_distances').fill_null([]),
    )
