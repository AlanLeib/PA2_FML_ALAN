from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from core import load_assets, read_measurements, transform, regroup_dbscan

st.set_page_config(page_title='PA2 | Explorador de clústeres', page_icon='🔬', layout='wide')
st.title('🔬 Explorador de características celulares')
st.caption('PA2 · Fundamentos de Machine Learning · K-Means y DBSCAN')
st.info('Herramienta académica de agrupamiento. Los resultados no determinan si una persona tiene o tendrá cáncer y no sustituyen una evaluación médica.')

@st.cache_resource
def assets():
    return load_assets(Path(__file__).resolve().parent)

try:
    km, db, scaler, columns = assets()
except Exception as exc:
    st.error('No se pudieron cargar los modelos y el preprocesamiento.')
    st.code(str(exc))
    st.markdown('Coloca `kmeans_model.pkl`, `dbscan_model.pkl` y `preprocesamiento.pkl` junto a `app.py`. Si hay incompatibilidad de versiones, utiliza la misma versión de scikit-learn que en Colab y reinicia la aplicación.')
    st.stop()

with st.sidebar:
    st.header('Preparar los datos')
    st.write('Carga un CSV con las mediciones originales, sin estandarizar. Los nombres deben coincidir con la plantilla.')
    st.download_button('Descargar plantilla CSV', pd.DataFrame(columns=columns).to_csv(index=False).encode(), 'plantilla_mediciones.csv', 'text/csv')
    st.caption('Una observación por fila. Decimales con punto. No incluyas nombres ni documentos personales.')
    sep = st.selectbox('Separador del CSV', [',', ';'])
    uploaded = st.file_uploader('Archivo de mediciones', type=['csv'])
    st.caption('Máximo 5 MB; 5000 filas para K-Means y 2000 para DBSCAN. El código de esta app no guarda los CSV en disco ni los publica en GitHub.')

with st.expander('¿Qué significan las variables y los resultados?'):
    st.write('Se utilizan las características seleccionadas en PA1. mean: promedio; se: error estándar; worst: promedio de los tres valores más extremos. Deben respetarse las unidades de la base de entrenamiento.')
    st.write('El número del clúster es una etiqueta arbitraria. Grupo 0 no significa negativo y grupo 1 no significa positivo. La relación con benigno/maligno se estudia después, contrastando grupos con diagnósticos conocidos; no permite predecir cáncer futuro.')
    st.write('K-Means utiliza el modelo entrenado. DBSCAN vuelve a agrupar el CSV cargado con los parámetros guardados: no dispone de predict nativo. Sus etiquetas pueden cambiar respecto del notebook.')
    st.write('Variables requeridas:')
    st.code('\n'.join(columns))

if uploaded is None:
    st.write('**Comienza descargando la plantilla y cargando tus mediciones desde la barra lateral.**')
    st.stop()
if uploaded.size > 5 * 1024 * 1024:
    st.error('El archivo supera los 5 MB.'); st.stop()
try:
    frame, extras = read_measurements(uploaded.getvalue(), columns, sep)
    values = transform(scaler, frame)
except Exception as exc:
    st.error('Revisa el archivo: ' + str(exc)); st.stop()
if extras:
    st.warning('Se excluyeron columnas adicionales. Solo se usan y exportan las mediciones requeridas.')
st.success(f'Archivo validado: {len(frame)} observaciones y {len(columns)} características.')
with st.expander('Vista previa de mediciones'):
    st.dataframe(frame.head(20), hide_index=True)
# Indicador descriptivo; no es validación del dominio ni un umbral clínico.
far = np.any(np.abs(values) > 5, axis=1)
if far.any():
    st.warning(f'{int(far.sum())} filas tienen alguna medición a más de 5 desviaciones estándar del promedio de entrenamiento. Revisa unidades y valores; este aviso es descriptivo.')


def results(labels, tag):
    labels = np.asarray(labels)
    n_groups = len(set(labels.tolist()) - {-1})
    a, b, c = st.columns(3)
    a.metric('Observaciones', len(labels)); b.metric('Grupos', n_groups)
    c.metric('Sin grupo / ruido', int((labels == -1).sum()))
    counts = pd.Series(labels, name='grupo').value_counts().sort_index()
    st.bar_chart(counts.rename('Observaciones'))
    result = frame.copy()
    result.insert(0, 'fila_csv', np.arange(2, len(frame) + 2))
    result['grupo'] = labels
    result['interpretacion'] = ['Ruido: sin asignación a un grupo denso' if v == -1 else f'Clúster {v}; no es un diagnóstico' for v in labels]
    st.dataframe(result, hide_index=True)
    st.download_button('Descargar resultados CSV', result.to_csv(index=False).encode('utf-8-sig'), f'resultados_{tag}.csv', 'text/csv', key=tag)
    st.caption('El ruido no significa malignidad; pertenecer a un grupo tampoco confirma ni descarta enfermedad.')

k_tab, d_tab = st.tabs(['K-Means · asignar grupos', 'DBSCAN · explorar densidad'])
with k_tab:
    st.subheader('Asignación con el K-Means entrenado')
    st.write(f'Cada fila se asigna al centroide más cercano de los {km.n_clusters} grupos aprendidos. Se conserva el escalador original.')
    try:
        results(km.predict(values), 'kmeans')
    except Exception as exc:
        st.error('No se pudo asignar el grupo: ' + str(exc))
with d_tab:
    st.subheader('Nuevo agrupamiento DBSCAN')
    st.write(f'Parámetros guardados: eps={db.eps:.4g}; min_samples={db.min_samples}. Se mantiene el escalador original y se ajusta una copia del algoritmo al CSV completo.')
    st.warning('Este proceso crea una nueva partición. No asigna casos individuales a los clústeres originales y no produce un resultado positivo/negativo de cáncer.')
    if len(frame) < db.min_samples:
        st.info('Hay menos filas que min_samples: todas las observaciones quedarán como ruido.')
    if len(frame) > 2000:
        st.error('DBSCAN admite hasta 2000 filas en esta aplicación.')
    elif st.checkbox('Ejecutar el nuevo agrupamiento sobre este CSV'):
        try:
            labels = regroup_dbscan(db, values)
            if len(set(labels) - {-1}) < 2:
                st.info('No se encontraron dos o más grupos densos. Esto es un resultado válido; no se fuerzan dos clases.')
            results(labels, 'dbscan')
        except Exception as exc:
            st.error('No se pudo agrupar: ' + str(exc))
