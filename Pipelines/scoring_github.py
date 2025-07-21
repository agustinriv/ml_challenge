# Librerías
import pandas as pd
import joblib
import os

from sklearn.metrics import r2_score

# Cargo los datos desde PostgreSQL
from sqlalchemy import create_engine

# Cargo datos directo de CSV
df = pd.read_csv('Dataset/dataset.csv')

# Tomo muestra aleatoria de 10 filas
df_sample = df.sample(n=10, random_state=42).reset_index(drop=True)
df_original = df_sample.copy()

# Columnas numéricas y categóricas
num_cols = ['age', 'bmi', 'children']
cat_cols = ['sex', 'smoker', 'region']

# Transformación variables categóricas

# Codifico usando get_dummies
df_encoded = pd.get_dummies(df_sample, columns=cat_cols)

# Cargo columnas esperadas desde training
columns_order = joblib.load("models/columns_order.pkl")

# Agrego columnas faltantes que estaban en entrenamiento
for col in columns_order:
    if col not in df_encoded.columns:
        df_encoded[col] = False

# Reordeno para que las columnas estén igual que en entrenamiento
df_encoded = df_encoded[columns_order]

# Cargo el modelo seleccionado
model = joblib.load("models/best_model_XGBoost.pkl")

# Predict
y_pred = model.predict(df_encoded)

# Mostrar resultados
df_original['predicted_charges'] = y_pred
print(df_original)

# R2
r2 = r2_score(df_original['charges'], df_original['predicted_charges'])
print(f"R2 sobre muestra de scoring: {r2:.4f}")

# Guardo resultados en csv
os.makedirs("scoring_output", exist_ok=True)
df_original.to_csv("scoring_output/scoring_results.csv", index=False)

