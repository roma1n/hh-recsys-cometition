import polars as pl

from hh import utils


@utils.timeit
def get_vacancies():
    return pl.read_parquet('data/hh_recsys_vacancies.pq')


@utils.timeit
def get_dssm():
    return pl.read_parquet('data/dssm_prediction.pq')


@utils.timeit
def get_train_hh():
    return pl.read_parquet('data/hh_recsys_train_hh.pq').with_columns(
        session_end=pl.col('action_dt').list.max(),
    )


@utils.timeit
def get_test_hh():
    return pl.read_parquet('data/hh_recsys_test_hh.pq').with_columns(
        session_end=pl.col('action_dt').list.max(),
    )


@utils.timeit
def get_vacancies_no_desc():
    return get_vacancies().select(pl.exclude('description', 'keySkills.keySkill'))


@utils.timeit
def get_log(training=False):
    log = pl.concat([
        get_train_hh(),
        get_test_hh(),
    ])
    if training:
        targets = pl.read_parquet('data/dssm_train.pq').select(
            pl.col('session_id'),
        ).unique()
        log = log.join(
            targets,
            on='session_id',
            how='anti',
        )
    return log


@utils.timeit
def get_targets():
    return get_log().select(
        pl.col('user_id'),
        pl.col('session_id'),
        pl.col('vacancy_id'),
        pl.col('action_type'),
        pl.col('session_end'),
    ).groupby(
        'user_id',
    ).agg(
        pl.all().sort_by('session_end').last(),
    ).explode(
        ['vacancy_id', 'action_type'],
    ).filter(
        pl.col('action_type') == 1,
    ).group_by(
        'user_id',
    ).agg(
        pl.col('vacancy_id').first().str.slice(2).cast(pl.UInt64).alias('target').add(1),
        pl.col('session_id').first(),
        pl.col('session_end').first(),
    )


def get_features(log, training=True):
    log = log.select(
        pl.col('user_id'),
        pl.col('session_id'),
        pl.col('vacancy_id'),
        pl.col('action_type'),
        pl.col('session_end'),
    ).groupby(
        'user_id',
    ).agg(
        pl.exclude('user_id').sort_by(
            'session_end',
        ),
        pl.count().alias('n_sessions'),
    )
    
    if training: # drop last session
        log = log.filter(
            pl.col('n_sessions') > 1
        ).select(
            [
                pl.col('user_id'),
                pl.col('n_sessions'),
            ] + [pl.col(c).list.head(
                    pl.col(c).list.len() - 1,
            ) for c in ['vacancy_id', 'action_type']],
        )

    return log.explode(
        ['vacancy_id', 'action_type'],
    ).explode(
        ['vacancy_id', 'action_type'],
    ).select(
        pl.exclude('vacancy_id'),
        pl.col('vacancy_id').str.slice(2).cast(pl.UInt64).add(1),
    ).group_by(
        'user_id',
        'n_sessions',
    ).agg(
        pl.col('vacancy_id').head(128),
        pl.col('action_type').head(128),
    ).select(
        pl.all(),
        pl.col('user_id').str.slice(2).cast(pl.UInt64).mod(7).eq(0).alias('is_test'),
    )


@utils.timeit
def get_train_features():
    return get_features(get_log())


@utils.timeit
def get_application_features():
    return get_features(get_log(), training=False)


@utils.timeit
def get_train_dataset():
    return get_train_features().join(
        get_targets(),
        on='user_id',
    )


@utils.timeit
def get_vacancy_features():
    name_mapping = {key: value + 2 for value, key in enumerate(get_vacancies_no_desc().group_by('name').count().filter(
        pl.col('count') > 10
    )['name'].to_list())}

    print('Name mapping len: ', len(name_mapping))

    return get_vacancies_no_desc().select(
        pl.col('vacancy_id').str.slice(2).cast(pl.UInt64).add(1),
        pl.col('area.regionId').str.slice(3).cast(pl.UInt64).add(2).fill_null(1),
        pl.col('area.id').str.slice(2).cast(pl.UInt64).add(1),
        pl.col('company.id').str.slice(2).cast(pl.UInt64).add(1),
        get_vacancies_no_desc()['workSchedule'].fill_null('fullDay').replace({
            "flexible": 1,
            "flyInFlyOut": 2,
            "shift": 3,
            "fullDay": 4,
            "remote": 5,
        }, return_dtype=pl.UInt64),
        get_vacancies_no_desc()['employment'].fill_null('full').replace({
            "full": 1,
            "project": 2,
            "volunteer": 3,
            "probation": 4,
            "part": 5,
        }, return_dtype=pl.UInt64),
        get_vacancies_no_desc()['workExperience'].fill_null('between1And3').replace({
            "between1And3": 1,
            "moreThan6": 2,
            "between3And6": 3,
            "noExperience": 4,
        }, return_dtype=pl.UInt64),
        get_vacancies_no_desc()['compensation.currencyCode'].fill_null('RUR').replace({
            "KGS": 1,
            "EUR": 2,
            "USD": 3,
            "AZN": 4,
            "BYR": 5,
            "UZS": 6,
            "UAH": 7,
            "GEL": 8,
            "KZT": 9,
            "RUR": 10,
        }, return_dtype=pl.UInt64),
        get_vacancies_no_desc()['name'].replace(name_mapping, return_dtype=pl.UInt64).fill_null(1),
    ).sort('vacancy_id')


@utils.timeit
def get_deleted_vacancies():  # DON'T use, works bad
    log = get_log(
        training=False, # dont use closed vacancies even in training
    ).select(
        'vacancy_id',
        'action_dt',
    ).explode(
        'vacancy_id',
        'action_dt',
    ).group_by(
        'vacancy_id'
    ).agg(
        pl.col('action_dt').cast(pl.Date).max(),
        pl.count().alias('n_action'),
    )
    last_log_dt = log['action_dt'].max()
    return log.filter(
        (
            pl.col('n_action') > 95  # at most 1% probability for random absence last date
        ) & (
            pl.col('action_dt') != last_log_dt
        ),
    )


@utils.timeit
def vacancy_action_stats():
    return get_log().select(  # always use full log
        'vacancy_id',
        'action_type',
        pl.col('action_dt').alias('dt'),
    ).explode(
        'vacancy_id',
        'action_type',
        'dt',
    ).select(
        'vacancy_id',
        pl.col('action_type').cast(pl.String),
        pl.col('dt').cast(pl.Date),
    ).pivot(
        index=[
            'vacancy_id',
            'dt',
        ],
        values=['action_type'],
        columns=['action_type'],
        aggregate_function='count',
    ).select(
        pl.all(),
        pl.col('1').add(pl.col('2')).add(pl.col('3')).alias('count'),
    ).sort('dt').select(
        pl.col('vacancy_id'),
        pl.col('dt'),
        pl.col('1').cum_sum().over('vacancy_id').alias('vacancy_action_1'),
        pl.col('2').cum_sum().over('vacancy_id').alias('vacancy_action_2'),
        pl.col('3').cum_sum().over('vacancy_id').alias('vacancy_action_3'),
        pl.col('count').cum_sum().over('vacancy_id').alias('vacancy_actions'),
        pl.col('count').alias('vacancy_actions_last_day'),
        (pl.col('count') / pl.col('count').cum_sum().over('vacancy_id')).alias('vacancy_actions_last_day_share'),
    ).sort('vacancy_id', 'dt')
