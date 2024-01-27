import numpy as np
import pandas as pd
import tqdm

class Candidate:
    def __init__(self, vac):
        self.vac = vac
        self.sessions_back_first = 1000
        self.sessions_back_last = 1000
        self.action_1 = 0
        self.action_2 = 0
        self.action_3 = 0

    def pass_session(self):
        self.sessions_back_last += 1
        self.sessions_back_first += 1

    def __repr__(self):
        return f'Candidate {self.vac}:\t{self.action_1}\t{self.action_2}\t{self.action_3}' \
            f'\t{self.sessions_back_first}\t{self.sessions_back_last}'

    def get_features(self):
        return {
            'vac': self.vac,
            'action_1': self.action_1,
            'action_2': self.action_2,
            'action_3': self.action_3,
            'sessions_back_first': self.sessions_back_first,
            'sessions_back_last': self.sessions_back_last,
        }

def candidates_to_df(d):
    return pd.DataFrame([e.get_features() for _, e in d.items()])


def get_dataset(input_path, output_path):
    # reed data
    data = pd.read_feather(input_path)
    data['session_end'] = data['action_dt'].apply(lambda l: max(l))

    # process data
    n_session = data.groupby('user_id').size().reset_index().rename(columns={0: 'n_session'})
    big_users = data.merge(n_session[(n_session['n_session'] > 1)][['user_id']], on='user_id')
    big_users = big_users.sort_values('session_end').groupby('user_id').tail(5)

    # run over users and prepare dataset
    dataset = []
    for user_id, user_sessions in tqdm.tqdm(big_users.groupby('user_id')):
        candidates = dict()
        for _, row in user_sessions.sort_values('session_end').iterrows():
            order = np.argsort(row['action_dt'])
            actions = row['action_type'][order]
            vacs = row['vacancy_id'][order]
            session = pd.DataFrame({
                'action': actions,
                'vac': vacs,
            })
        
            applies = session[session['action'] == 1][['vac']]
            applies['target'] = 1
        
            if len(candidates) > 0:
                candidates_df = candidates_to_df(candidates)
                candidates_df = candidates_df.merge(applies, how='left', on='vac')
                candidates_df['target'] = candidates_df['target'].fillna(0)
                candidates_df = pd.concat([
                    candidates_df[candidates_df['target'] == 1],
                    candidates_df[candidates_df['target'] == 0].sample(frac=0.05),
                ])
                candidates_df['user_id'] = user_id
                candidates_df['session_id'] = row['session_id']
                candidates_df['session_end'] = row['session_end']

                dataset.append(candidates_df)
        
            # add new candidates
            for _, action in session.iterrows():
                if action['vac'] not in candidates:
                    candidates[action['vac']] = Candidate(action['vac'])
                    candidates[action['vac']].sessions_back_first = 0
                candidates[action['vac']].sessions_back_last = 0
        
                if action['action'] == 1:
                    candidates[action['vac']].action_1 += 1
                elif action['action'] == 2: 
                    candidates[action['vac']].action_2 += 1
                elif action['action'] == 3: 
                    candidates[action['vac']].action_3 += 1
        
            # update candidates
            for vac in candidates:
                candidates[vac].pass_session()

    # save dataset
    dataset = pd.concat(dataset, ignore_index=True)
    dataset.to_feather(output_path)
