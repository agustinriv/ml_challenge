# MetLife Challenge

## Contenido

### Pipelines
- training.py: pipeline de entrenamiento, lee los datos desde PostgreSQL
- training_github.py: versión alternativa de training el cual lee los datos desde archivo CSV
- scoring.py

### Notebooks
- eda.ipynb: exploración de datos inicial
- training.ipynb: desarrollo del pipeline de entrenamiento
- scoring.ipynb: desarrollo del pipeline de scoring
- setup.ipynb: carga del dataset a PostgreSQL

### Archivos generados por los pipelines
- best_model_XGBoost.pkl: modelo entrenado con mejor desempeño (menor RMSE). Se usa en scoring.py para hacer predicciones
- columns_order.pkl: lista del orden exacto de las columnas luego de aplicar pd.get_dummies(). Permite al pipeline de scoring alinear correctamente las columnas del nuevo dataset con las usadas en el entrenamiento
- resultados_modelos.txt: resultados de evaluación de cada modelo probado (Random Forest, Gradient Boosting, XGBoost), incluyendo hiperparámetros, RMSE, y R² para el conjunto de test
- scoring_results.csv: contiene 10 filas del dataset original con una nueva columna predicted_charges, generada por el modelo entrenado. También se calcula y muestra el R² sobre esta muestra

## Procedimiento y Resultados

Durante el desarrollo del proyecto se probaron tres algoritmos de regresión: Random Forest, Gradient Boosting Regressor y XGBoost Regressor, utilizando búsqueda de hiperparámetros mediante GridSearchCV y validación cruzada KFold (5 folds).
Como métrica de evaluación se utilizaron RMSE (Root Mean Squared Error) y R² (coeficiente de determinación), calculados sobre un conjunto de test del 20%.

El modelo con mejor desempeño fue XGBoost, con un R² de 0.8817 y un RMSE de 4285.22.

## Guía para ejecutar pipelines

1) Crear un entorno virtual e instalar dependencias
2) Subir dataset a PostgreSQL. Ejecutar setup.ipynb para crear la tabla en la base de datos local
3) Ejecutar el script de entrenamiento desde la terminal (python Pipelines/training.py)
4) Ejecutar el script de entrenamiento desde la terminal (python Pipelines/scoring.py)
5) En caso de preferir no utilizar PostgreSQL, hay dos pipelines alternativos que toman los datos directamente del csv guardado en el repositorio. Estos pipelines se encuentran como training_github.py y scoring_github.py

## Consideraciones finales

### Testeo

Los pipelines fueron testeados en un entorno de desarrollo a través de GitHub Codespaces, lo cual permitió verificar que tanto training.py como scoring.py se ejecutan sin errores y generan las salidas esperadas (modelos entrenados, métricas, predicciones).

### Docker Image

Si bien no se implementó una imagen Docker, el enfoque podría ser:

Crear un archivo Docker File en el repositorio que contenga las instrucciones necesarias para  contruir un Docker Image del proyecto y luego correrlo. Este contenedor tendría los archivos del proyecto, tendría instaladas las dependencias listadas en requirements, y estaria configurado para que corra primero el pipeline de entrenamiento y luego el de scoring.

Crear un Docker Image permitiria ejecutar el proceso completo de forma reproducible, independientement del entorno local del usuario.


### Conclusiones

Si bien se llegó a un resultado concreto, algunas acciones para mejorar el modelo serían:

- Realizar una búsqueda más extensa de hiperparámetros con más recursos de cómputo
- Incorporar mayor volumen de datos: el dataset actual es relativamente pequeño
- Aplicar técnicas de Feature Engineering para generar nuevas variables o transformar las existentes
- Explorar otros algoritmos y analizar sus resultados
