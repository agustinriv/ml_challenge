# Librerías
import pandas as pd
import joblib
import os

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score

# Cargo los datos desde PostgreSQL
from sqlalchemy import create_engine

# Parámetros de conexión
db_user = 'agustinrivas'
db_host = 'localhost'
db_port = '5432'
db_name = 'dataset_ml'

engine = create_engine(f'postgresql://{db_user}@{db_host}:{db_port}/{db_name}')
df = pd.read_sql('SELECT * FROM training_dataset', engine)

# Tomo muestra aleatoria de 10 filas
df_sample = df.sample(n=10, random_state=42).reset_index(drop=True)
df_original = df_sample.copy()

# Columnas numéricas y categóricas
num_cols = ['age', 'bmi', 'children']
cat_cols = ['sex', 'smoker', 'region']

# Transformación variables categóricas

encoder = OneHotEncoder(drop='first', sparse_output = False)
encoder.fit(df[cat_cols])

columns_encoded = encoder.get_feature_names_out(cat_cols)
x_encoded = encoder.transform(df_sample[cat_cols])
encoded_df = pd.DataFrame(x_encoded, columns=columns_encoded, index=df_sample.index)
# Agrego columnas faltantes con 0 (en caso de que haya)
for col in columns_encoded:
    if col not in encoded_df.columns:
        encoded_df[col] = 0

# Reordeno columnas
encoded_df = encoded_df[columns_encoded]

# Concateno con num_cols
df_final = pd.concat([encoded_df, df_sample[num_cols]], axis=1)

# Cargo el modelo seleccionado
model = joblib.load("models/best_model_XGBoost.pkl")

# Predict
y_pred = model.predict(df_final)

# Mostrar resultados
df_original['predicted_charges'] = y_pred
print(df_original)

# R2
r2 = r2_score(df_original['charges'], df_original['predicted_charges'])
print(f"R2 sobre muestra de scoring: {r2:.2f}")

# Guardo resultados en csv
os.makedirs("scoring_output", exist_ok=True)
df_original.to_csv("scoring_output/scoring_results.csv", index=False)

