import hnswlib
import numpy as np
import polars as pl
import pytorch_lightning as li 
from pytorch_lightning.callbacks import EarlyStopping
import torch
from torch import nn

from hh import utils


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

EMBED_DIM = 64
VACANCY_ID_EMBED_DIM = 32
COMPANY_ID_EMBED_DIM = 16
AREA_ID_EMBED_DIM = 16
AREA_REGION_ID_EMBED_DIM = 8
EMPLOYMENT_EMBED_DIM = 4
WORK_SCHEDULE_EMBED_DIM = 4
WORK_EXPERIENCE_EMBED_DIM = 4
COMPENSATION_CURRENCY_CODE_EMBED_DIM = 4
NAME_EMBED_DIM = 16

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
N_EPOCH = 5
BATCH_SIZE = 32

NUM_WORKERS = 4


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


class EmbedSingleVac(nn.Module):
    def __init__(self, description):
        super().__init__()

        self.description = description

        self.vacancy_id_embed = nn.Embedding(
            num_embeddings=N_VACANCY_ID,
            embedding_dim=VACANCY_ID_EMBED_DIM,
        )
        self.company_id = nn.Embedding(
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
        self.fc = nn.Linear(
            in_features=VACANCY_ID_EMBED_DIM + COMPANY_ID_EMBED_DIM + AREA_ID_EMBED_DIM + AREA_REGION_ID_EMBED_DIM \
                + EMPLOYMENT_EMBED_DIM + WORK_SCHEDULE_EMBED_DIM + WORK_EXPERIENCE_EMBED_DIM \
                + COMPENSATION_CURRENCY_CODE_EMBED_DIM + NAME_EMBED_DIM,
            out_features=EMBED_DIM,
        )

    def forward(self, x):
        x = torch.cat([
            self.vacancy_id_embed(x),
            self.company_id(self.description.company_id[x]),
            self.area_id(self.description.area_id[x]),
            self.area_region_id(self.description.area_region_id[x]),
            self.work_schedule(self.description.work_schedule[x]),
            self.employment(self.description.employment[x]),
            self.work_experience(self.description.work_experience[x]),
            self.compensation_currency_code(self.description.compensation_currency_code[x]),
            self.name(self.description.name[x]),
        ], dim=1)
        x = self.fc(x)
        return x


class EmbedMultipleVac(nn.Module):
    def __init__(self, description):
        super().__init__()

        self.description = description

        self.vacancy_id_embed = nn.EmbeddingBag(
            num_embeddings=N_VACANCY_ID,
            embedding_dim=VACANCY_ID_EMBED_DIM,
        )
        self.company_id = nn.EmbeddingBag(
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
        self.fc = nn.Linear(
            in_features=VACANCY_ID_EMBED_DIM + COMPANY_ID_EMBED_DIM + AREA_ID_EMBED_DIM + AREA_REGION_ID_EMBED_DIM \
                + EMPLOYMENT_EMBED_DIM + WORK_SCHEDULE_EMBED_DIM + WORK_EXPERIENCE_EMBED_DIM \
                + COMPENSATION_CURRENCY_CODE_EMBED_DIM + NAME_EMBED_DIM,
            out_features=EMBED_DIM,
        )

    def forward(self, x):
        x = torch.cat([
            self.vacancy_id_embed(x),
            self.company_id(self.description.company_id[x]),
            self.area_id(self.description.area_id[x]),
            self.area_region_id(self.description.area_region_id[x]),
            self.work_schedule(self.description.work_schedule[x]),
            self.employment(self.description.employment[x]),
            self.work_experience(self.description.work_experience[x]),
            self.compensation_currency_code(self.description.compensation_currency_code[x]),
            self.name(self.description.name[x]),
        ], dim=1)
        x = self.fc(x)
        return x


class DSSM(li.LightningModule):
    def __init__(self, embed_x, embed_y):
        super().__init__()
        self.embed_x = embed_x
        self.embed_y = embed_y
        self.dropout = nn.Dropout1d(p=0.2)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, y):
        x = self.embed_x(x)
        x = self.dropout(x)
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
        loss = self.criterion(logits, torch.eye(batch_size))
        # loss = (
        #     self.criterion(logits, torch.eye(batch_size)) \
        #         + self.criterion(logits.T, torch.eye(batch_size))
        # ) / 2
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
            loss = self.criterion(logits, torch.eye(batch_size))
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
                patience=5,
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
    train = pl.read_parquet('data/dssm_train.pq')
    targets = np.array(train['target'].to_list())

    # Drop targets that had low occurency while training
    ts, cs = np.unique(targets, return_counts=True)
    goodts = ts[cs >= 2]
    p = hnswlib.Index(space='ip', dim=EMBED_DIM)
    p.init_index(max_elements=goodts.shape[0], ef_construction=400, M=16)
    p.add_items(vacancy_embeddings[goodts], goodts)
    return p


def get_predictions_by_index(p, user_embeddings, users_df):
    labels, distances = p.knn_query(user_embeddings, k=300)
    labels = np.char.add(np.str_("v_"), (labels - 1).astype(str))
    return pl.DataFrame().with_columns(
        users_df['user_id'],
        dssm=labels,
        dssm_distances=distances,
    ).select(
        pl.col('dssm'),
        pl.col('dssm_distances'),
        pl.col('user_id').cast(pl.String),
    )


@utils.timeit
def get_predictions(path='data/user_application_features.pq'):
    description = VacancyDescription(path='data/vacancy_features.pq')
    dssm = DSSM.load_from_checkpoint(
        'data/epoch=1-step=13269.ckpt',
        embed_x=EmbedMultipleVac(description=description),
        embed_y=EmbedSingleVac(description=description),
    )
    users_df = pl.read_parquet(path).group_by('user_id').agg(
        pl.col('vacancy_id').first(),
    )
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
