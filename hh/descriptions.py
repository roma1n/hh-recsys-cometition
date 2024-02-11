import numpy as np
import tqdm
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from pandarallel import pandarallel

from hh import data


def get_text(row):
    skills = row['keySkills.keySkill']
    skills = skills.tolist() if isinstance(skills, list) else []
    text = '. '.join(
        [
            row['name'],
            BeautifulSoup(row['description'], features='html.parser').get_text(),
        ] + skills,
    )[:1024]
    return text.strip()


def calculate():
    vacancies = data.get_vacancies()
    
    print('Calculating vacancy texts')
    pandarallel.initialize(
        nb_workers=16,
    )
    vacancies = vacancies.to_pandas()
    vacancies['text'] = vacancies.parallel_apply(get_text, axis='columns')
    
    print('Initialising model')
    model = SentenceTransformer('cointegrated/rubert-tiny2')
    
    print('Calculating embeddings')
    parts = []
    for chunk in tqdm.tqdm(np.array_split(vacancies, vacancies.shape[0] // 1000)):
        r = model.encode(chunk['text'].tolist())
        parts.append(np.array(r))
    full = np.concatenate(parts)
    
    fname = 'data/vac_text.npz'
    print('Saving embeddings to ', fname)
    np.savez(fname, full)
