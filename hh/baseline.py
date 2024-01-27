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
    return candidates.get_application_candidates().select(
        pl.col('user_id'),
        pl.concat_list([
            pl.col('likes').list.set_difference('applies'),
            pl.col('views').list.set_difference('applies'),
            pl.col('dssm').list.set_difference('applies'),
        ]).list.unique(maintain_order=True).list.head(100).alias('predictions'),
    )
