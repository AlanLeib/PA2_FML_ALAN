# PA2 · Aplicación Streamlit

Una app con dos pestañas, carga de CSV y descarga de resultados. K-Means asigna nuevas mediciones a los centroides guardados. DBSCAN crea una nueva partición del CSV utilizando los parámetros guardados, sin modificar el objeto original. No realiza diagnósticos ni predice cáncer futuro.

## Archivos que debes agregar

Coloca en la misma carpeta de app.py los tres archivos que exportaste con joblib:

- kmeans_model.pkl: objeto KMeans entrenado.
- dbscan_model.pkl: objeto DBSCAN entrenado.
- preprocesamiento.pkl: diccionario con claves `scaler` y `columnas`, conservando el orden original.

Los modelos del usuario no están incluidos en este paquete. No se han podido probar sus PKL concretos. Solo se cargan PKL locales del repositorio; los visitantes únicamente pueden subir CSV.

## Compatibilidad de versiones

En el notebook que utilizaste para exportar los PKL ejecuta:

```python
import sklearn, numpy, pandas, joblib
print('scikit-learn==' + sklearn.__version__)
print('numpy==' + numpy.__version__)
print('pandas==' + pandas.__version__)
print('joblib==' + joblib.__version__)
```

En requirements.txt reemplaza las líneas correspondientes por esas salidas. Usa preferentemente la misma versión de Python del entrenamiento. Los rangos incluidos son una plantilla inicial, no una garantía de compatibilidad con tus PKL. La app detiene la carga si scikit-learn reporta una versión incompatible.

## Subir a GitHub y desplegar

1. Descomprime este paquete.
2. Sube app.py, core.py, requirements.txt y tus tres PKL a la carpeta principal de tu repositorio. No subas el ZIP como único archivo. No publiques datos personales ni CSV de pacientes.
3. En Streamlit Community Cloud, crea una aplicación desde tu repositorio de GitHub.
4. Selecciona la rama correspondiente y `app.py` como archivo principal; configura la versión de Python cuando corresponda.
5. Despliega. Si cambias los PKL, reinicia la app para limpiar los objetos en caché.
6. Descarga la plantilla desde la app y carga un CSV de mediciones originales, sin escalar. Usa punto decimal; separador coma o punto y coma.

Documentación: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## Ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Interpretación

Los números de clúster no equivalen a benigno/maligno. No se incluyen asociaciones clínicas fijas porque los archivos concretos del usuario y su correspondencia con diagnósticos no han sido verificados. DBSCAN etiqueta el ruido con -1. Una asignación a grupo es una operación geométrica, no una probabilidad de enfermedad. Para clasificar benigno/maligno se necesitaría un clasificador supervisado validado, como el de PA1, manteniendo esa función separada del clustering.

## Validación

Se probaron carga joblib con modelos de prueba, orden de columnas, rechazos de datos incompletos y no numéricos, compatibilidad de dimensiones, asignación K-Means, y nuevo agrupamiento DBSCAN sin modificar el modelo guardado. La prueba no establece validez clínica ni reemplaza la validación con tus propios PKL.
