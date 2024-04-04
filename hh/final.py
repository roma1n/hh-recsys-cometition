import catboost
import datetime
import matplotlib.pyplot as plt
import polars as pl
import shap
import tqdm

from hh import (
    candidates,
    data,
    utils,
)


eq_features = [
    'name',
    'company.id',
    'area.id',
    'employment',
    'workSchedule',
    'workExperience',
]
float_features = list(map(lambda feature: f'eq_{feature}', eq_features)) + [
    'likes_count',
    'applies_count',
    'views_count',
    'likes_back',
    'applies_back',
    'views_back',
    'score',
    'score_pos',
    'fm_score',
    'fm_score_pos',
    'sessions_back',
    'vacancy_actions',
    'vacancy_actions_last_day',
    'vacancy_actions_last_day_share',
    'vacancy_since_action',
    'vacancy_action_1',
    'vacancy_action_2',
    'vacancy_action_3',
    # 'compensation.from',
    # 'compensation.to',
]
cat_features = [
    'vacancy_id',
    # 'name',
    # 'company.id',
    # 'area.id',
    # 'employment',
    # 'workSchedule',
    # 'workExperience',
    # 'compensation.currencyCode',
    # 'area.regionId',
]
ranker_features = float_features + cat_features


@utils.timeit
def precalculate_dataset(training=True):
    if training:
        targets =  data.get_log().select(
            pl.col('user_id'),
            pl.col('vacancy_id'),
            pl.col('action_type'),
            pl.col('session_end'),
        ).groupby(
            'user_id',
        ).agg(
            pl.all().sort_by('session_end').last(),
            pl.count(),
        ).filter(
            pl.col('count') > 1,
        ).explode(
            ['vacancy_id', 'action_type'],
        ).filter(
            pl.col('action_type') == 1,
        ).group_by(
            'user_id',
        ).agg(
            pl.col('vacancy_id').alias('target'),
            pl.col('session_end').first(),
        ).select(
            pl.col('user_id'),
            pl.col('target'),
            pl.col('session_end').cast(pl.Date).alias('dt'),
            pl.col('user_id').str.slice(2).cast(pl.UInt64).mod(7).eq(0).alias('is_test'),
        )
        dataset = targets.join(
            candidates.get_application_candidates(training=training),
            on='user_id',
        )
        return dataset
    return candidates.get_application_candidates(training=training).with_columns(
        dt=pl.lit(datetime.date(year=2023, month=11, day=22))
    )


@utils.timeit
def get_dataset(training=True):
    pre_dataset = precalculate_dataset(training=training)
    views, applies, likes = [pre_dataset.select(
        pl.col('user_id'),
        pl.col(action),
        pl.col(action).list.len().map_elements(
            lambda s: list(range(s)),
            return_dtype=pl.List(pl.Int64),
        ).alias(f'{action}_back'),
    ).explode(
        action,
        f'{action}_back',
    ).filter(
        pl.col(action).is_not_null(),
    ).group_by(
        'user_id',
        pl.col(action).alias('vacancy_id'),
    ).agg(
        pl.count().alias(f'{action}_count'),
        pl.col(f'{action}_back').min(),
    ) for action in ['views', 'applies', 'likes']]

    dssm = pre_dataset.select(
        pl.col('user_id'),
        pl.col('dssm').alias('vacancy_id'),
        pl.col('dssm_distances').alias('score'),
        pl.lit(list(range(300))).alias('score_pos'),
    ).explode(
        'vacancy_id',
        'score',
        'score_pos',
    )
    fm = pre_dataset.select(
        pl.col('user_id'),
        pl.col('fm').alias('vacancy_id'),
        pl.col('fm_distances').alias('fm_score'),
        pl.lit(list(range(300))).alias('fm_score_pos'),
    ).explode(
        'vacancy_id',
        'fm_score',
        'fm_score_pos',
    )
    dataset = likes.join(
        applies,
        on=['user_id', 'vacancy_id'], 
        how='outer_coalesce',
    ).join(
        views,
        on=['user_id', 'vacancy_id'], 
        how='outer_coalesce',
    ).join(
        dssm,
        on=['user_id', 'vacancy_id'], 
        how='outer_coalesce',
    ).join(
        fm,
        on=['user_id', 'vacancy_id'], 
        how='outer_coalesce',
    ).select(
        pl.col(['user_id']),
        pl.col(['vacancy_id']),
        pl.col(['likes_count']).fill_null(0),
        pl.col(['applies_count']).fill_null(0),
        pl.col(['views_count']).fill_null(0),
        pl.col(['likes_back']).fill_null(1000),
        pl.col(['applies_back']).fill_null(1000),
        pl.col(['views_back']).fill_null(1000),
        pl.col(['score']).fill_null(10),
        pl.col(['score_pos']).fill_null(1000),
        pl.col(['fm_score']).fill_null(10),
        pl.col(['fm_score_pos']).fill_null(1000),
    ).join(
        data.get_vacancies_no_desc(),
        on='vacancy_id',
    ).join(
        pre_dataset.select(
            'user_id',
            'dt',
        ),
        on='user_id',
    )

    print('Joining vacancy action features')
    vacancy_action_features = dataset.select(
        pl.col('dt').alias('dataset_dt'),
        'vacancy_id',
    ).unique().join(
        data.vacancy_action_stats(),
        on='vacancy_id',
    ).filter(
        pl.col('dt') < pl.col('dataset_dt')  # stats before dataset row
    ).sort('dt', descending=True).group_by(
        'vacancy_id', 'dataset_dt'
    ).first().select(
        pl.exclude(['dt', 'dataset_dt']),
        pl.col('dataset_dt').alias('dt'),
        pl.col('dataset_dt').sub(pl.col('dt')).dt.total_days().alias('vacancy_since_action'),
    )
    dataset = dataset.join(
        vacancy_action_features,
        on=['vacancy_id', 'dt'],
        how='left',
    ).select(
        pl.exclude([
            'vacancy_actions',
            'vacancy_actions_last_day',
            'vacancy_actions_last_day_share',
            'vacancy_since_action',
            'vacancy_action_1',
            'vacancy_action_2',
            'vacancy_action_3',
        ]),
        pl.col('vacancy_actions').fill_null(0),
        pl.col('vacancy_actions_last_day').fill_null(0),
        pl.col('vacancy_actions_last_day_share').fill_null(1),
        pl.col('vacancy_since_action').fill_null(30),
        pl.col('vacancy_action_1').fill_null(0),
        pl.col('vacancy_action_2').fill_null(0),
        pl.col('vacancy_action_3').fill_null(0),
    )

    print('Building flog')
    flog = data.get_log(training=training).select(
        'user_id',
        'vacancy_id',
    ).explode(
        'vacancy_id',
    ).join(
        data.get_vacancies_no_desc(),
        on='vacancy_id',
    )
    print('Calculating eqs')
    eqs = [flog.group_by(
        pl.col('user_id'),
        pl.col(feature),
    ).count().select(
        pl.exclude('count'),
        pl.col('count').alias(f'eq_{feature}')
    ) for feature in eq_features]
    print('Joining eq features')
    for feature, eq in zip(eq_features, eqs):
        dataset = dataset.join(
            eq,
            on=['user_id', feature],
            how='left',
        ).select(
            pl.exclude(f'eq_{feature}'),
            pl.col(f'eq_{feature}').fill_null(0),
        )

    print('Calculating sessions_back')
    last_sessions = data.get_log(training=training).sort('session_end', descending=True).group_by(
        'user_id'
    ).agg(
        pl.col('session_end'),
        pl.cum_count().alias('sessions_back'),
    ).select(
        'user_id',
        pl.col('session_end').list.head(10).alias('last_sessions'),
        pl.col('sessions_back').list.head(10),
    )
    sessions_back = data.get_log(training=training).select(
        'user_id',
        'session_end',
        'vacancy_id',
    ).explode(
        'vacancy_id',
    ).join(
        last_sessions,
        on='user_id'
    ).explode(
        'sessions_back',
        'last_sessions',
    ).filter(
        pl.col('session_end') == pl.col('last_sessions'),
    ).group_by(
        'user_id',
        'vacancy_id'
    ).agg(
        pl.col('sessions_back').min(),
    )
    dataset = dataset.join(
        sessions_back,
        on=['user_id', 'vacancy_id'],
        how='left',
    ).select(
        pl.exclude('sessions_back'),
        pl.col('sessions_back').fill_null(20),
    )

    if training:
        dataset = dataset.join(
            pre_dataset.select(
                'user_id',
                'target',
                'is_test',
            ),
            on=['user_id'], 
            how='inner',
        ).select(
            pl.exclude('target'),
            pl.col('vacancy_id').is_in(pl.col('target')).cast(pl.Int64).alias('target'),
        )
        sample = pl.concat([
            dataset.filter(
                pl.col('target') == 1
            ),
            dataset.filter(
                pl.col('target') == 0
            ).sample(fraction=0.1),
        ])
        return sample
    return dataset


def train_catboost():
    sample = pl.read_parquet('data/final_train_dataset.pq').with_columns(
        ts=pl.col('dt').dt.timestamp()
    )
    df = sample.to_pandas()

    df[cat_features] = df[cat_features].astype(str)
    df[float_features] = df[float_features].astype(float)

    train = df[~df.is_test].sort_values('user_id')
    test = df[df.is_test].sort_values('user_id')
    print(f'Split sizes: train - {train.shape[0]}, test - {test.shape[0]}')

    train_pool = catboost.Pool(
        data=train[ranker_features],
        label=train['target'],
        cat_features=cat_features,
        group_id=train['user_id'],
        timestamp=train['ts'],
    )
    test_pool = catboost.Pool(
        data=test[ranker_features],
        label=test['target'],
        cat_features=cat_features,
        group_id=test['user_id'],
         timestamp=test['ts'],
    )
    model = catboost.CatBoostRanker(
        n_estimators=1000,
        eval_metric='MRR:top=100',
    )
    model.fit(
        X=train_pool,
        eval_set=test_pool,
        verbose=10,
        early_stopping_rounds=100,
    )
    model.save_model('data/model.cbm')


@utils.timeit
def get_predictions():
    data = pl.read_parquet('data/final_application_dataset.pq')
    batch_size = 1_000_000
    scores = []
    model = catboost.CatBoostRanker().load_model('data/model.cbm')
    for batch in tqdm.tqdm(data.iter_slices(n_rows=batch_size), total=len(data) // batch_size + 1):
        batch = batch
        scores.append(pl.select(
            batch['user_id'],
            batch['vacancy_id'],
            pl.Series(model.predict(batch.to_pandas()[ranker_features])).alias('score')
        ))
    scores = pl.concat(scores)
    return scores.sort('score', descending=True).group_by(
        'user_id'
    ).agg(
        pl.col('vacancy_id').alias('predictions'),
    ).select(
        pl.col('user_id'),
        pl.col('predictions').list.head(100),
    )


def catboost_shap():
    sample = pl.read_parquet('data/final_application_dataset.pq').head(10_000).to_pandas()
    model = catboost.CatBoostRanker().load_model('data/model.cbm')
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample[ranker_features])

    shap.summary_plot(
        shap_values,
        feature_names=ranker_features,
        features=sample[ranker_features],
        plot_size=(8, 12),
        show=False,
    )
    plt.savefig('catboost_info/shap.png')
