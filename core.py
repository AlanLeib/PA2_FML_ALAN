"""Carga de modelos locales y validación de mediciones."""
from pathlib import Path
import csv
import io
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.utils.validation import check_is_fitted


def load_assets(folder):
    folder = Path(folder)
    with warnings.catch_warnings():
        warnings.simplefilter('error', InconsistentVersionWarning)
        pre = joblib.load(folder / 'preprocesamiento.pkl')
        km = joblib.load(folder / 'kmeans_model.pkl')
        db = joblib.load(folder / 'dbscan_model.pkl')
    if not isinstance(pre, dict) or not {'scaler', 'columnas'} <= set(pre):
        raise ValueError('preprocesamiento.pkl debe contener scaler y columnas.')
    columns = list(pre['columnas'])
    if not columns or not all(isinstance(c, str) for c in columns) or len(set(columns)) != len(columns):
        raise ValueError('La lista de columnas no es válida.')
    if any('diagnos' in c.lower() or c.lower() in {'target', 'outcome'} for c in columns):
        raise ValueError('El diagnóstico no debe ser una variable de entrada.')
    if not isinstance(km, KMeans) or not isinstance(db, DBSCAN):
        raise ValueError('Los archivos deben contener modelos KMeans y DBSCAN individuales.')
    for model in (km, db, pre['scaler']):
        check_is_fitted(model)
        if getattr(model, 'n_features_in_', None) != len(columns):
            raise ValueError('El número de variables del escalador y los modelos no coincide.')
    if hasattr(pre['scaler'], 'feature_names_in_'):
        if list(pre['scaler'].feature_names_in_) != columns:
            raise ValueError('El orden de columnas no coincide con el escalador.')
    return km, db, pre['scaler'], columns


def read_measurements(raw, columns, separator=','):
    text = raw.decode('utf-8-sig')
    header = next(csv.reader(io.StringIO(text), delimiter=separator), [])
    if len(header) != len(set(header)):
        raise ValueError('El CSV tiene encabezados duplicados.')
    frame = pd.read_csv(io.StringIO(text), sep=separator)
    if not 1 <= len(frame) <= 5000:
        raise ValueError('Carga entre 1 y 5000 filas.')
    missing = [c for c in columns if c not in frame]
    if missing:
        raise ValueError('Faltan columnas: ' + ', '.join(missing))
    extra = [c for c in frame if c not in columns]
    selected = frame.loc[:, columns].apply(pd.to_numeric, errors='raise')
    values = selected.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Hay celdas vacías, infinitos o valores no válidos en las mediciones.')
    if (values < 0).any():
        raise ValueError('Las mediciones de esta base no pueden ser negativas.')
    return selected, extra


def transform(scaler, frame):
    values = scaler.transform(frame)
    if not np.isfinite(values).all():
        raise ValueError('El escalado produjo valores no válidos.')
    return values


def regroup_dbscan(saved_model, values):
    """Nuevo ajuste independiente; no modifica el modelo guardado."""
    if len(values) > 2000:
        raise ValueError('Para DBSCAN carga como máximo 2000 filas.')
    model = DBSCAN(**saved_model.get_params())
    return model.fit_predict(values)
