# HH Recsys

## Как запустить обучение:

1. Варим датасет для dssm:
```bash
hh dssm_train_dataset
```

2. Обучаем dssm
```bash
hh train_dssm
```

3. Обучаем катбуст
```bash
hh final_train_dataset
hh final_train_catboost
```

## Как запустить предсказание:

1. Строим предсказание dssm

```bash
hh dssm_prediction
```

2. Джойним предсказания

```bash
hh history_plus_dssm
```

3. Применяем катбуст

```bash
hh final_application_dataset
hh final_get_predictions
```
