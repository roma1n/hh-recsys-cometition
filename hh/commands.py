import fire
import polars as pl

from hh import (
    baseline,
    data,
    dssm,
    final,
    utils,
)


class HeadHunter:
    def __init__(self):
        print ('Running hh')

    def history_baseline(self):
        utils.save_pq(
            baseline.get_history_baseline(),
            'data/history_baseline.pq',
        )

    def history_plus_dssm(self):
        utils.save_pq(
            baseline.get_history_plus_dssm(),
            'data/history_plus_dssm.pq',
        )

    def get_vacancy_features(self):
        utils.save_pq(
            data.get_vacancy_features(),
            'data/vacancy_features.pq'
        )

    def dssm_train_dataset(self):
        '''
        Prepare dssm training dataset (no vacancy features).
        Mathod: take last session for each users for targets, previous sessions are taket for features
        '''
        utils.save_pq(
            data.get_train_dataset(),
            'data/dssm_train.pq'
        )

    def user_application_features(self):
        utils.save_pq(
            data.get_application_features(),
            'data/user_application_features.pq'
        )

    def dssm_prediction(self):
        utils.save_pq(
            dssm.get_predictions(),
            'data/dssm_prediction.pq'
        )

    def dssm_test_prediction(self):
        path = 'data/dssm_train.pq'
        utils.save_pq(
            dssm.get_predictions(path),
            'data/dssm_test_prediction.pq'
        )

    def train_dssm(self):
        dssm.train()
    
    def final_train_dataset(self):
        utils.save_pq(
            final.get_dataset(),
            'data/final_train_dataset.pq'
        )

    def final_application_dataset(self):
        utils.save_pq(
            final.get_dataset(training=False),
            'data/final_application_dataset.pq'
        )

    def final_train_catboost(self):
        final.train_catboost()

    def final_get_predictions(self):
        utils.save_pq(
            final.get_predictions(),
            'data/catboost_predictions.pq'
        )

def main():
    fire.Fire(HeadHunter)
