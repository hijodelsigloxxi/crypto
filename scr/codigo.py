import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, mean_squared_error, r2_score

#Validacion cruzada
df= pd.read_csv('criptomonedas_historico_consolidad.csv')

features = ['open', 'high', 'low', 'volume', 'sma_50', 'ema_20', 'rsi', 'macd', 'bollinger_upper', 'bollinger_lower', 'roc']
X = df[features]
y = df['close']

imputador = SimpleImputer(strategy='mean')
X_imputed= pd.DataFrame(imputador.fit_transform)
X_imputed.columns=X.columns

model=RandomForestRegressor(n_estimators=100, random_state=42)

kf= KFold(n_splits=5, shuffle=True, random_state=42)


mse_scorer= make_scorer(mse_score, greater_is_better=False)
r2_score= make_scorer(r2_score_custom, greater_is_better=True)

mse_scores = cross_val_score(model, X_imputed, y, scoring=mse_scorer, cv=kf)
r2_scores = cross_val_score(model, X_imputed, y, scoring=r2_scorer, cv=kf)

mse_scores = -mse_scores
print("MSE scores:", mse_scores)
print("Mean MSE:", mse_scores.mean())
print("R² scores:", r2_scores)
print("Mean R²:", r2_scores.mean())







df_train = pd.read_csv("criptomonedas_historico_consolidado.csv")

df_test = pd.read_csv("criptomonedas_historico_consolidado_semana.csv")


features = ['open', 'high', 'low', 'volume', 'sma_50', 'ema_20', 'rsi', 'macd', 'bollinger_upper', 'bollinger_lower', 'roc']
X_train = df_train[features]
y_train = df_train['close']


imputador = SimpleImputer(strategy='mean')
X_train_imputed = pd.DataFrame(imputador.fit_transform(X_train))
X_train_imputed.columns = X_train.columns

# Preprocesamiento de datos para la prueba
X_test = df_test[features]
y_test = df_test['close']

# Imputar los NaNs para el conjunto de prueba
X_test_imputed = pd.DataFrame(imputador.transform(X_test))
X_test_imputed.columns = X_test.columns

# Definir el modelo
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Entrenar el modelo con los datos de entrenamiento
model.fit(X_train_imputed, y_train)

# Predecir con los datos de prueba
y_pred = model.predict(X_test_imputed)

# Calcular métricas de rendimiento
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE) en los datos de prueba:", mse)
print("Coeficiente de Determinación (R²) en los datos de prueba:", r2)



