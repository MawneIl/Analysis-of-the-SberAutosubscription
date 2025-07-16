import dill
import os
import pandas as pd
import logging
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

from sklearn.model_selection import cross_val_score

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task


""""
Версия 2.0
Автор: Nail Mavliev
Model: AutoML
Тип: классификация
Дата: 2025-07-15
"""

# variables:
path = os.environ.get('PROJECT_PATH', '.')

target_actions = ['sub_car_claim_click',
                  'sub_car_claim_submit_click',
                  'sub_open_dialog_click',
                  'sub_custom_question_submit_click',
                  'sub_call_number_click',
                  'sub_callback_submit_click',
                  'sub_submit_success',
                  'sub_car_request_submit_click'
                  ]

TIMEOUT = 3600
CPU_LIMIT = 10
CV = 5
RS = 42

logging.basicConfig(level=logging.INFO, filename=f"{path}/logs/project_log_v2.log", filemode="a",
                    format="%(asctime)s %(levelname)s %(message)s")

class LightAutoMLWrapper(BaseEstimator, ClassifierMixin):
    """Обертка для TabularAutoML для совместимости с sklearn Pipeline"""
    def __init__(self, task, timeout, cpu_limit, reader_params, roles):
        self.task = task
        self.timeout = timeout
        self.cpu_limit = cpu_limit
        self.reader_params = reader_params
        self.roles = roles
        self.model = None
        
    def fit(self, X, y):
        # Преобразуем данные в pandas DataFrame, если это необходимо
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        # объединяем X и y для обучения, так как TabularAutoML ожидает DataFrame с target
        if 'target' not in X.columns:
            X['target'] = y      

        self.model = TabularAutoML(
            task=self.task,
            timeout=self.timeout,
            cpu_limit=self.cpu_limit,
            reader_params=self.reader_params,
        )
        self.model.fit_predict(X, roles=self.roles)
        return self
        
    def predict_proba(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return self.model.predict(X).data[:, 0]
            
    def predict(self, X):
        probability = self.predict_proba(X)
        # Преобразуем вероятности в бинарные метки 
        return (probability >= 0.5).astype(int)
    
    def _model_name(self):
        if hasattr(self.model, '_model_name'):
            return self.model._model_name
        return "TabularAutoML"


def df_load():
    files = ['ga_hits.pkl', 'ga_sessions.pkl']
    with open(f'{path}/data/{files[0]}', 'rb') as file:
        df_hits = dill.load(file)

    with open(f'{path}/data/{files[1]}', 'rb') as file:
        df_sessions = dill.load(file)

    df_hits['target'] = df_hits.event_action.apply(
        lambda x: 1 if x in target_actions else 0)
    pivot_table = pd.pivot_table(df_hits, index=['session_id'], values=['target'],
                                 aggfunc={'target': [
                                     lambda x: 0 if x.sum() == 0 else 1]}
                                 ).reset_index(level=0)
    pivot_table.columns = ['session_id', 'target']

    df = df_sessions.join(pivot_table.set_index(
        'session_id'), on='session_id', how='inner')

    df = df[df.utm_source.notna()]

    return df

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    organic_traffic = ['organic', 'referral', '(none)']

    social_media = ['QxAxdyPLuQMEcrdZWdWb', 'MvfHsxITijuriZxsqZqt',
                    'ISrKoXQCxqqYvAZICvjs', 'IZEXUFLARCUMynmHNBGo',
                    'PlbkrSYoHuZBWfYjYnfw', 'gVRrcxiDQubJiljoTbGm'
                    ]

    df['utm_social_media'] = df.apply(
        lambda x: 1 if x.utm_source in social_media else 0, axis=1)

    df['utm_traffic'] = df.apply(
        lambda x: 'organic' if x.utm_medium in organic_traffic else 'paid', axis=1)

    df['geo_from_russia'] = df.apply(
        lambda x: 1 if x.geo_country == "Russia" else 0, axis=1)

    return df

def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = {
        'session_id',
        'client_id',
        'visit_date',
        'visit_time',
    }

    columns_to_save = list(set(df.columns.to_list()) - columns_to_drop)

    return df[columns_to_save]

def pipeline():

    print('Target event predictor Pipeline')
    logging.info('Target event predictor Pipeline')

    print('Loading data...')
    logging.info('Loading data...')

    df = df_load()

    x = df.drop('target', axis=1)
    y = df['target']

    print(f"Nan y: {y.isna().sum()} from {len(y)}")

    print('Preparing data...')
    logging.info('Preparing data...')

    # numerical_features = make_column_selector(
    #     dtype_include=['int64', 'float64'])
    # categorical_features = make_column_selector(dtype_include=object)

    # numerical_transformer = Pipeline(steps=[
    #     ('imputer', SimpleImputer(strategy='median')),
    #     ('scaler', StandardScaler())
    # ])

    # categorical_transformer = Pipeline(steps=[
    #     ('imputer', SimpleImputer(strategy='constant',
    #                               fill_value='other')),
    #     ('encoder', OneHotEncoder(handle_unknown='ignore'))
    # ])

    # column_transformer = ColumnTransformer(
    #     transformers=[
    #     ('numerical', numerical_transformer, numerical_features),
    #     ('categorical', categorical_transformer, categorical_features)
    #     ])

    preprocessor = Pipeline(steps=[
        ('feature_creator', FunctionTransformer(create_features)),
        ('filter', FunctionTransformer(filter_data)),
        # ('column_transformer', column_transformer)    # Убираем, так как TabularAutoML сам обрабатывает данные и ожидает DataFrame
    ])

    print('Creating model...')
    logging.info('Creating model...')

    task = Task('binary', metric='auc')
    roles = {
        'target': 'target',
        'drop': ['session_id', 'client_id', 'visit_date', 'visit_time'],
    }

    # Используем обертку вместо прямого использования TabularAutoML
    model = LightAutoMLWrapper(
            task=task,
            timeout=TIMEOUT,
            cpu_limit=CPU_LIMIT,
            reader_params={'n_jobs': CPU_LIMIT, 'cv': CV, 'random_state': RS},
            roles=roles)

    print('Fitting model...')
    logging.info('Fitting model...')

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    pipe.fit(x, y)

    print('Model training completed. Model evaluation...')
    logging.info('Model training completed. Model evaluation...')

    rocauc = roc_auc_score(y, pipe.predict_proba(x))
    accuracy = accuracy_score(pipe.predict(x), y)

    accuracy = accuracy_score(pipe.predict(x), y)
    logging.info(f'model: {pipe["classifier"]._model_name()}, '
                 f'roc_auc: {rocauc:.4f}, '
                 f'accuracy: {accuracy:.2f}')

    print('Saving model...')
    logging.info('Saving model...')

    model_filename = f'{path}/models/{pipe["classifier"]._model_name()}_v2.pkl'

    with open(model_filename, 'wb') as file:
        dill.dump({
            'model': pipe,
            'metadata': {
                'name': 'Target event prediction pipeline',
                'autor': 'Nail Mavliev',
                'version': 2.0,
                'date': datetime.now(),
                'type': pipe["classifier"]._model_name(),
                'roc_auc': rocauc,
                'accuracy': accuracy
            }
        }, file)

    logging.info(f'Model is saved as {model_filename}')

if __name__ == '__main__':
    pipeline()