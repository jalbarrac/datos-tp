import pandas as pd
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn import model_selection
from sklearn.preprocessing import (
    KBinsDiscretizer,
    LabelEncoder,
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)



def droppear_nulos_por_columna(data):
    NULL_REMOVE_PCT = 0.30
    cols = data.isna().mean()
    cols = cols[cols < NULL_REMOVE_PCT]
    return data[cols.index]

def droppear_filas_sin_sede(data):
    _data = data.drop(data.loc[data['nombre_sede'].isna()].index, inplace=False)
    _data.reset_index(drop=True)
    return _data

def droppear_columnas_con_valores_unicos(X):
    _X = X.copy(deep=True)
    _X.drop(['id_ticket','nombre','id_usuario'], axis=1, inplace=True)
    return _X

def input_edad_median(X):
    _X = X.copy(deep=True)
    _X['edad'] = SimpleImputer(strategy='median').fit_transform(_X[['edad']])
    return _X

def encode_onehot(X):
    _X = X.copy(deep=True)
    columns = ['tipo_de_sala', 'genero', 'nombre_sede']
    encoder = OneHotEncoder(drop='first')
    for col in columns:
            
        encoded_data = encoder.fit(_X[[col]].astype(str))
        encoded_data_categories = list(encoded_data.categories_)
        
        encoded_data = encoded_data.transform(_X[[col]].astype(str)).todense().astype(int)
        encoded_data = pd.DataFrame(encoded_data)
        
        encoded_data_categories = np.delete(encoded_data_categories, 0)
        encoded_data.columns = encoded_data_categories
            
        _X = pd.concat([X, encoded_data], axis=1)
        _X.drop(labels=col, axis=1, inplace=True)
        X = _X
    
    return _X

def scale_minmax():
    pass

#llamar con 0.2 y 7
def split(X, y, size, state):
    
    _X = X.copy(deep=True)
    _y = y.copy(deep=True)
    
    X_train, X_test, y_train, y_test = model_selection.train_test_split(_X, _y, test_size=size, random_state=state, stratify=y['volveria'])
    
    y_train = y_train['volveria'].to_numpy(copy=True)
    y_test = y_test['volveria'].to_numpy(copy=True)
    
    return X_train, X_test, y_train, y_test

def prepro_base(X, y, train):
    _X = X.copy(deep=True)
    _X = droppear_nulos_por_columna(_X)
    _X = droppear_columnas_con_valores_unicos(_X)
    _X = encode_onehot(_X)
    _X = input_edad_median(_X)
    
    if(train):
        X_train, X_test, y_train, y_test = split(_X, y, 0.2, 13)
        return X_train, X_test, y_train, y_test
    
    return _X



scaling_params = {'standard_withmean':True}
selection_params = {'vt_threshold':0,
                    'rfe_estimator':'estimator'}

#scalers
def StandardScalerWrapper(scaling_params):
    return StandardScaler(with_mean=scaling_params['standard_withmean'])

def MinMaxScalerWrapper(scaling_params):
    return MinMaxScaler()

def RobustScalerWrapper(scaling_params):
    return RobustScaler()

def PowerTransformerWrapper(scaling_params):
    return PowerTransformer()

def NormalizerWrapper(scaling_params):
    return Normalizer()

scalers = {'standard': StandardScalerWrapper(scaling_params),
          'minmax': MinMaxScalerWrapper(scaling_params),
          'robust': RobustScalerWrapper(scaling_params),
           'power' : PowerTransformerWrapper(scaling_params),
           'normalizer' : NormalizerWrapper(scaling_params)
          }

#Selectors


def VarianceThresholdWrapper(selection_params):
    return VarianceThreshold()

def RFEWrapper(selection_params):
    return RFE(selection_params['rfe_estimator'])

def FeatureHasherWrapper(selection_params):
    return FeatureHasher()


selectors = {'var_thres': VarianceThresholdWrapper(selection_params),
            'rfe': RFEWrapper(selection_params),
            'feature_hasher': FeatureHasherWrapper(selection_params)}
