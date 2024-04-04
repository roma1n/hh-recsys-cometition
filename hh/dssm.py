import hnswlib
import numpy as np
import polars as pl
import pytorch_lightning as li
from pytorch_lightning.callbacks import EarlyStopping
import torch
from torch import nn

from hh import (
    data,
    utils,
)


HISTORY_LENGTH = 64
N_VACANCY_ID = 2734130
N_COMPANY_ID = 278913
N_AREA_ID = 7015
N_AREA_REGION_ID = 107
N_EMPLOYMENT = 6
N_WORK_SCHEDULE = 6
N_WORK_EXPERIENCE = 5
N_COMPENSATION_CURRENCY_CODE = 11
N_NAME = 18060

EMBED_DIM = 256
DESCRIPTION_EMBED_DIM = 312
VACANCY_ID_EMBED_DIM = 0
COMPANY_ID_EMBED_DIM = 16
AREA_ID_EMBED_DIM = 32
AREA_REGION_ID_EMBED_DIM = 8
EMPLOYMENT_EMBED_DIM = 4
WORK_SCHEDULE_EMBED_DIM = 4
WORK_EXPERIENCE_EMBED_DIM = 4
COMPENSATION_CURRENCY_CODE_EMBED_DIM = 4
NAME_EMBED_DIM = 16

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-6
N_EPOCH = 100
BATCH_SIZE = 128

NUM_WORKERS = 16


class VacancyDescription():
    def __init__(
        self,
        path='data/vacancy_features.pq',
    ):
        self.vacancies = pl.read_parquet(path).sort('vacancy_id')
        self.company_id = torch.cat([torch.tensor([0]), torch.tensor(self.vacancies['company.id'].to_list())])
        self.area_id = torch.cat([torch.tensor([0]), torch.tensor(self.vacancies['area.id'].to_list())])
        self.area_region_id = torch.cat([torch.tensor([0]), torch.tensor(self.vacancies['area.regionId'].to_list())])
        self.work_schedule = torch.cat([torch.tensor([0]), torch.tensor(self.vacancies['workSchedule'].to_list())])
        self.employment = torch.cat([torch.tensor([0]), torch.tensor(self.vacancies['employment'].to_list())])
        self.work_experience = torch.cat([torch.tensor([0]), torch.tensor(self.vacancies['workExperience'].to_list())])
        self.compensation_currency_code = torch.cat([torch.tensor([0]), torch.tensor(self.vacancies['compensation.currencyCode'].to_list())])
        self.name = torch.cat([torch.tensor([0]), torch.tensor(self.vacancies['name'].to_list())])

        self.name = self.name.clip(max=N_NAME - 1) # fix strange big names

        self.text = torch.zeros(N_VACANCY_ID, DESCRIPTION_EMBED_DIM, dtype=torch.float32)
        text_np = np.load('data/vac_text.npz')['arr_0']
        order = data.get_vacancies().select(
            pl.col('vacancy_id').str.slice(2).cast(pl.UInt64).add(1),
        ).to_pandas()['vacancy_id'].tolist()
        self.text[order] = torch.tensor(text_np)


class EmbedSingleVac(nn.Module):
    @staticmethod
    def _get_fc_block(
        in_features,
        out_features,
    ):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Dropout1d(p=0.2),
            nn.ReLU(),
            nn.Linear(out_features, out_features),
            nn.ReLU(),
            nn.Dropout1d(p=0.2),
        )

    def __init__(
            self,
            description,
            company_id=None,
        ):
        super().__init__()

        self.description = description
        self.company_id = company_id or nn.EmbeddingBag(
            num_embeddings=N_COMPANY_ID,
            embedding_dim=COMPANY_ID_EMBED_DIM,
        )
        self.area_id = nn.Embedding(
            num_embeddings=N_AREA_ID,
            embedding_dim=AREA_ID_EMBED_DIM,
        )
        self.area_region_id = nn.Embedding(
            num_embeddings=N_AREA_REGION_ID,
            embedding_dim=AREA_REGION_ID_EMBED_DIM,
        )
        self.employment = nn.Embedding(
            num_embeddings=N_EMPLOYMENT,
            embedding_dim=EMPLOYMENT_EMBED_DIM,
        )
        self.work_schedule = nn.Embedding(
            num_embeddings=N_WORK_SCHEDULE,
            embedding_dim=WORK_SCHEDULE_EMBED_DIM,
        )
        self.work_experience = nn.Embedding(
            num_embeddings=N_WORK_EXPERIENCE,
            embedding_dim=WORK_EXPERIENCE_EMBED_DIM,
        )
        self.compensation_currency_code = nn.Embedding(
            num_embeddings=N_COMPENSATION_CURRENCY_CODE,
            embedding_dim=COMPENSATION_CURRENCY_CODE_EMBED_DIM,
        )
        self.name = nn.Embedding(
            num_embeddings=N_NAME,
            embedding_dim=NAME_EMBED_DIM,
        )
        self.fc = self._get_fc_block(
            in_features=VACANCY_ID_EMBED_DIM + COMPANY_ID_EMBED_DIM + AREA_ID_EMBED_DIM + AREA_REGION_ID_EMBED_DIM \
                + EMPLOYMENT_EMBED_DIM + WORK_SCHEDULE_EMBED_DIM + WORK_EXPERIENCE_EMBED_DIM \
                + COMPENSATION_CURRENCY_CODE_EMBED_DIM + NAME_EMBED_DIM + DESCRIPTION_EMBED_DIM,
            out_features=EMBED_DIM,
        )
        self.fc_2 = self._get_fc_block(EMBED_DIM, EMBED_DIM)
        self.fc_3 = self._get_fc_block(EMBED_DIM, EMBED_DIM)
        self.fc_4 = self._get_fc_block(EMBED_DIM, EMBED_DIM)
        self.fc_out = nn.Linear(4 * EMBED_DIM, EMBED_DIM)

    def forward(self, x):
        x = torch.cat([
            self.company_id(self.description.company_id[x].unsqueeze(1)),
            self.area_id(self.description.area_id[x]),
            self.area_region_id(self.description.area_region_id[x]),
            self.work_schedule(self.description.work_schedule[x]),
            self.employment(self.description.employment[x]),
            self.work_experience(self.description.work_experience[x]),
            self.compensation_currency_code(self.description.compensation_currency_code[x]),
            self.name(self.description.name[x]),
            self.description.text[x],
        ], dim=1)
        x = self.fc(x)
        y = self.fc_2(x)
        z = self.fc_3(y)
        p = self.fc_4(z)
        return self.fc_out(torch.cat([x, y, z, p], dim=1))


class EmbedMultipleVac(nn.Module):
    @staticmethod
    def _get_fc_block(
        in_features,
        out_features,
    ):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features),
            nn.ReLU(),
        )

    def __init__(
            self,
            description,
            company_id=None,
        ):
        super().__init__()

        self.description = description
        self.company_id = company_id or nn.EmbeddingBag(
            num_embeddings=N_COMPANY_ID,
            embedding_dim=COMPANY_ID_EMBED_DIM,
        )
        self.area_id = nn.EmbeddingBag(
            num_embeddings=N_AREA_ID,
            embedding_dim=AREA_ID_EMBED_DIM,
        )
        self.area_region_id = nn.EmbeddingBag(
            num_embeddings=N_AREA_REGION_ID,
            embedding_dim=AREA_REGION_ID_EMBED_DIM,
        )
        self.employment = nn.EmbeddingBag(
            num_embeddings=N_EMPLOYMENT,
            embedding_dim=EMPLOYMENT_EMBED_DIM,
        )
        self.work_schedule = nn.EmbeddingBag(
            num_embeddings=N_WORK_SCHEDULE,
            embedding_dim=WORK_SCHEDULE_EMBED_DIM,
        )
        self.work_experience = nn.EmbeddingBag(
            num_embeddings=N_WORK_EXPERIENCE,
            embedding_dim=WORK_EXPERIENCE_EMBED_DIM,
        )
        self.compensation_currency_code = nn.EmbeddingBag(
            num_embeddings=N_COMPENSATION_CURRENCY_CODE,
            embedding_dim=COMPENSATION_CURRENCY_CODE_EMBED_DIM,
        )
        self.name = nn.EmbeddingBag(
            num_embeddings=N_NAME,
            embedding_dim=NAME_EMBED_DIM,
        )
        self.fc = self._get_fc_block(
            in_features=VACANCY_ID_EMBED_DIM + COMPANY_ID_EMBED_DIM + AREA_ID_EMBED_DIM + AREA_REGION_ID_EMBED_DIM \
                + EMPLOYMENT_EMBED_DIM + WORK_SCHEDULE_EMBED_DIM + WORK_EXPERIENCE_EMBED_DIM \
                + COMPENSATION_CURRENCY_CODE_EMBED_DIM + NAME_EMBED_DIM + DESCRIPTION_EMBED_DIM,
            out_features=EMBED_DIM,
        )
        self.fc_2 = self._get_fc_block(EMBED_DIM, EMBED_DIM)
        self.fc_3 = self._get_fc_block(EMBED_DIM, EMBED_DIM)
        self.fc_4 = self._get_fc_block(EMBED_DIM, EMBED_DIM)
        self.fc_out = nn.Linear(4 * EMBED_DIM, EMBED_DIM)

    def forward(self, x):
        x = torch.cat([
            self.company_id(self.description.company_id[x]),
            self.area_id(self.description.area_id[x]),
            self.area_region_id(self.description.area_region_id[x]),
            self.work_schedule(self.description.work_schedule[x]),
            self.employment(self.description.employment[x]),
            self.work_experience(self.description.work_experience[x]),
            self.compensation_currency_code(self.description.compensation_currency_code[x]),
            self.name(self.description.name[x]),
            self.description.text[x].mean(dim=1),
        ], dim=1)
        x = self.fc(x)
        y = self.fc_2(x)
        z = self.fc_3(y)
        p = self.fc_4(z)
        return self.fc_out(torch.cat([x, y, z, p], dim=1))


class DSSM(li.LightningModule):
    def __init__(self, embed_x, embed_y):
        super().__init__()
        self.embed_x = embed_x
        self.embed_y = embed_y
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, y):
        x = self.embed_x(x)
        y = self.embed_y(y)

        return (x[:,None,:] * y[None,:,:]).sum(dim=2) # pairwise batch multiplication

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

    def training_step(self, batch, batch_idx):
        'Trains model on batch'
        self.train(True)
        x, y = batch
        batch_size = x.shape[0]
        logits = self.forward(x, y)
        loss = 0.5 * (self.criterion(logits, torch.eye(batch_size)) + self.criterion(logits.T, torch.eye(batch_size)))
        self.log(
            'train_loss',
            loss.item(),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

        predictions = logits.detach().cpu().argmax(axis=1)
        self.log(
            'train_accuracy',
            (predictions == torch.arange(batch_size)).float().mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        'Validates model on batch'
        self.train(False)
        x, y = batch
        batch_size = x.shape[0]
        with torch.no_grad():
            logits = self.forward(x, y)
            loss = 0.5 * (self.criterion(logits, torch.eye(batch_size)) + self.criterion(logits.T, torch.eye(batch_size)))
            self.log(
                'val_loss',
                loss.item(),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
            )

            predictions = logits.detach().cpu().argmax(axis=1)
            self.log(
                'val_accuracy',
                (predictions == torch.arange(batch_size)).float().mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            return loss


class HeadHunterDataModule(li.LightningDataModule):
    def __init__(self):
        super().__init__()

        self.train_dataset, self.val_dataset = self.get_train_val_datasets()

    @staticmethod
    def pad(seqs):
        def pad_seq(seq):
            seq = seq[:HISTORY_LENGTH]
            return seq + [0] * (HISTORY_LENGTH - len(seq))
        return [pad_seq(seq) for seq in seqs]

    def get_train_val_datasets(self):
        print('Loading train-val dataset')
        full = pl.read_parquet('data/dssm_train.pq')
        train = full.filter(~pl.col('is_test'))
        val = full.filter(pl.col('is_test'))
        print(f'Train samples: {train.shape[0]}; val samples: {val.shape[0]}')

        return (
            torch.utils.data.TensorDataset(
                torch.tensor(self.pad(train['vacancy_id'].to_list()), dtype=torch.long),
                torch.tensor(train['target'].to_list(), dtype=torch.long),
            ),
            torch.utils.data.TensorDataset(
                torch.tensor(self.pad(val['vacancy_id'].to_list()), dtype=torch.long),
                torch.tensor(val['target'].to_list(), dtype=torch.long),
            ),
        )

    def make_data_loader(self, dataset, shuffle=False):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            num_workers=NUM_WORKERS,
            persistent_workers=True,
        )

    def train_dataloader(self):
        return self.make_data_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self.make_data_loader(self.val_dataset)


def train():
    description = VacancyDescription()
    print('Training DSSM')
    trainer = li.Trainer(
        precision='bf16-mixed',
        max_epochs=N_EPOCH,
        log_every_n_steps=100,
        logger=[
            li.loggers.CSVLogger('logs'),
        ],
        val_check_interval=100,
        callbacks=[
            EarlyStopping(
                monitor='val_loss',
                min_delta=0.001,
                patience=100,
            ),
        ],
    )
    dssm = DSSM(
        embed_x=EmbedMultipleVac(description=description),
        embed_y=EmbedSingleVac(description=description),
    )
    datamodule = HeadHunterDataModule()
    trainer.fit(
        model=dssm,
        datamodule=datamodule,
    )


def get_vacancy_embeddings(dssm):
    return dssm.embed_y(
        torch.arange(0, dssm.embed_y.description.company_id.shape[0], dtype=torch.long),
    ).detach().cpu().numpy()


def get_user_embeddings(dssm, users_df):
    users_tensor = torch.tensor(
        HeadHunterDataModule.pad(users_df['vacancy_id'].to_list()),
        dtype=torch.long,
    )
    return dssm.embed_x(users_tensor).detach().cpu().numpy()


def build_index(vacancy_embeddings):
    goodts = np.arange(1, N_VACANCY_ID)
    p = hnswlib.Index(
        space='cosine',
        dim=EMBED_DIM,
    )
    p.init_index(max_elements=goodts.shape[0], ef_construction=1000, M=64)
    p.add_items(vacancy_embeddings[goodts], goodts)
    return p


def get_predictions_by_index(p, user_embeddings, users_df):
    labels, distances = p.knn_query(user_embeddings, k=300)
    return pl.DataFrame().with_columns(
        users_df['user_id'],
        dssm=labels,
        dssm_distances=distances,
    ).explode(
        'dssm_distances',
        'dssm',
    ).select(
        pl.concat_str(pl.lit('v_'), pl.col('dssm').sub(1).cast(pl.String)).alias('dssm'),
        pl.col('dssm_distances'),
        pl.col('user_id').cast(pl.String),
    ).group_by(
        'user_id',
    ).agg(
        pl.col('dssm'),
        pl.col('dssm_distances'),
    )


@utils.timeit
def get_predictions(path='data/user_application_features.pq'):
    description = VacancyDescription(path='data/vacancy_features.pq')
    dssm = DSSM.load_from_checkpoint(
        'data/epoch=78-step=57830.ckpt',
        embed_x=EmbedMultipleVac(description=description),
        embed_y=EmbedSingleVac(description=description),
    )
    dssm.train(False)
    users_df = pl.read_parquet(path)
    with torch.no_grad():
        return get_predictions_by_index(
            p=build_index(
                vacancy_embeddings=get_vacancy_embeddings(
                    dssm=dssm,
                ),
            ),
            user_embeddings=get_user_embeddings(
                dssm=dssm,
                users_df=users_df,
            ),
            users_df=users_df,
        )
