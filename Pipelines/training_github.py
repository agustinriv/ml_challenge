# Librerías
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

from xgboost import XGBRegressor

# Cargo datos directamente del csv
df = pd.read_csv("dataset/dataset.csv")

# Variable Target
X = df.drop('charges', axis=1)
y = df['charges']

# Columnas numéricas y categóricas
num_cols = ['age', 'bmi', 'children']
cat_cols = ['sex', 'smoker', 'region']

# Transformación variables categóricas
df_final = pd.get_dummies(X, columns = cat_cols)

# Orden de columnas para scoring
columns_order = df_final.columns.tolist()
joblib.dump(columns_order, 'models/columns_order.pkl')

# Split de los datos
X_train, X_test, y_train, y_test = train_test_split(df_final, y, test_size=0.2, random_state=42)

# Grilla para modelos: RandomForest, GradientBoostingRegressor, XGBoost
grid_models = {
    'RandomForest': {
        'model': RandomForestRegressor(random_state=42),
        'param': {
            'n_estimators': [100, 200],
            'max_depth': [1,5,10,25,50],
            'min_samples_leaf':[2,25,50],
            'min_samples_split':[2,25,50],
        }
    },
    'GradientBoosting': {
        'model': GradientBoostingRegressor(random_state=42),
        'param': {
            'n_estimators': [100, 200],
            'learning_rate': [0.1, 0.05],
            'max_depth': [1,5,10,25,50],
        }
    },
    'XGBoost': {
        'model': XGBRegressor(random_state=42),
        'param': {
            'n_estimators': [100, 200],
            'learning_rate': [0.1, 0.05],
            'max_depth': [1,5,10,25,50],
        }
    }
}

# Obtengo el mejor modelo con sus hiperparámetros
results = []
cv = KFold(n_splits=5, shuffle=True, random_state=42)

for modelo, config in grid_models.items():

    grid = GridSearchCV(
        estimator=config['model'],
        param_grid=config['param'],
        cv=cv,
        scoring='neg_mean_squared_error',
        n_jobs=1
    )
    
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    results.append((modelo,grid.best_params_, rmse,r2))

# Convierto resultados en DataFrame
df_results = pd.DataFrame(results, columns=['Modelo', 'params', 'RMSE', 'R2'])

# Entre los mejores modelos, selecciono el de menor rmse
min_rmse = df_results.loc[df_results['RMSE'].idxmin()]
best_model_name = min_rmse['Modelo']

# Entreno mejor modelo
best_model = grid_models[best_model_name]['model'].set_params(**min_rmse['params'])
best_model.fit(X_train, y_train)

# Creao carpeta y guardo modelo
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, f"models/best_model_{best_model_name}.pkl")

# Creo carpeta y guardo métricas en .txt
os.makedirs("metrics",exist_ok=True)
with open("metrics/resultados_modelos.txt", "w") as f:
    f.write("Evaluación de modelos\n")
    f.write(df_results.to_string(index=False))
