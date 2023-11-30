print(" ")
print('En este apartado se simularan secuencias de los activos financieros para entrenar la red recurrente y valorar su eficacia:')
print(" ")
print('0- LIBRERIAS')
print("Cargando librerias necesarias...")
# LIBRERIAS
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats.mstats import gmean
from numpy.linalg import cholesky
from scipy.stats import norm
import os
import random
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import packages.Modelado.funciones_modelado as mdl
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from scipy.stats import norm
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Establecemos semilla
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

# Creamos carpetas necesarias
if not os.path.exists("Models_checkpoint"):
    os.makedirs('Models_checkpoint')

# 1- PREPARACIÓN DE LOS DATOS
print(" ")
# Cargamos datos de los 3 activos
print('1- CARGA  Y PREPARACIÓN DE DATOS')
print('Estos son los datos que se emplearan en el modelado:')
print('Serie temporal del precio "close" del activo JPM. Directorio ->"Datos/Transformados/diarios/JPM_marketstack.csv" ')
print('Serie temporal del precio "close" del activo CAT. Directorio ->"Datos/Transformados/diarios/CAT_marketstack.csv" ')
print('Serie temporal del precio "close" del activo GS. Directorio ->"Datos/Transformados/diarios/GS_marketstack.csv" ')
df1 = pd.read_csv('Datos/Transformados/diarios/JPM_marketstack.csv')
df2 = pd.read_csv('Datos/Transformados/diarios/CAT_marketstack.csv')
df3 = pd.read_csv('Datos/Transformados/diarios/GS_marketstack.csv')
df = pd.concat([df1['close_cubic'], df2['close_cubic'], df3['close_cubic']], axis=1)
df.columns = ['JPM', 'CAT', 'GS']

# Preparamos variables para emplear montecarlo
df_corr = df.corr()
rho = df_corr.values 
m = cholesky(rho)
S_0 = np.array([df['JPM'][0],df['CAT'][0],df['GS'][0]])
num_steps = 20 
vols = np.array([0.332834,0.303271,0.301934]).reshape(-1,1) 
risk_free_rate = 0.1
TTM = initial_maturity = 5 
value_date_index = 5 
num_asian_dates = num_steps + 1
init_time_array = np.linspace(0, initial_maturity, num_asian_dates)
correl_matrix = rho.copy()
K = 3
num_sims = 100000
num_assets = 3 

# 2- MONTECARLO
print(" ")
print('2- MONTECARLO')
# Aplicamos montecarlo
print('2- Aplicando Montecarlo para simular los datos y crear carteras...')
rets, payoff, premium = mdl.MonteCarlo(initial_maturity, S_0, K=3, num_sims=num_sims, num_assets=num_assets, num_asian_dates=num_asian_dates, value_date_index=value_date_index, correl_matrix=correl_matrix, risk_free_rate=risk_free_rate, vols=vols)

# 3- MODELADO
print(" ")
print("3- MODELADO")
# Separamos los datos en train, valiacion y test
test_ratio = 0.05
X_train_full,X_test = train_test_split(rets,test_size=test_ratio,random_state=42)
y_train_full,y_test = train_test_split(payoff,test_size=test_ratio,random_state=42)
X_train,X_val = X_train_full[:90000,:,:],X_train_full[90000:95000,:,:]
Y_train,Y_val = y_train_full[:90000],y_train_full[90000:95000]
print(f'Se entrenará el modelo con {X_train.shape[0]} simulaciones, el {(1-test_ratio*2)*100}% de los datos ')
print(f'La validación y el testeo de la eficacia del modelo se realizarán con {X_test.shape[0]} carteras, el {test_ratio*100}% de los datos')

print("Entrenando modelo con los datos generados...")
# Definimos el modelo
model=mdl.get_model()
model=mdl.train_model(model, X_train, Y_train, X_val, Y_val)
# Predecimos los valores de test y mostramos el error
y_pred = model.predict(X_test)
model_mse,model_mae=mdl.model_test(y_test,y_pred)
print(" ")
print('Estos son los errores que comete el modelo en la predicción de los datos de test:')
print(f'MSE: {model_mse}')
print(f'MAE: {model_mae}')


# 4- COMPARACIÓN CON FORMULA CERRADA BBVA
print(" ")
print('4- COMPARACIÓN RESPECTO BBVA')
print('Comparamos las predicciones de nuestro modelo con las de la formula cerrada de BBVA para las primas de montecarlo variando el numero de simulaciones y los pasos empleados en la simulación:')
# Definimos los valores
#df_corr = df.corr()
#rho = df_corr.values 
rho=rho.tolist()
#vols = np.array([0.332834,0.303271,0.301934]).reshape(-1,1) 
#risk_free_rate = 0.1
#value_date_index = 5 
#num_asian_dates = num_steps + 1 
#init_time_array = np.linspace(0, initial_maturity, num_asian_dates)
initial_maturity=20
#num_assets = 3 
N = norm.cdf

# Cargamos el modelo
model=keras.models.load_model("./Models_checkpoint/model.hdf5")
print('Realizando simulación...')
predicciones=mdl.predicciones_finales(model,rets,initial_maturity,S_0,num_sims,num_assets,num_asian_dates,correl_matrix,risk_free_rate,vols,rho,TTM,N)
resultados=mdl.calculos_errores(predicciones)
# montecarlo_long=resultados.groupby("long").mean()
# montecarlo_nsim=resultados.groupby("n_sim").mean()
medias=np.mean(resultados[['MAE_BBVA','MAE_RED_RECURRENTE','MSE_BBVA','MSE_RED_RECURRENTE']], axis=0)
print("Errores que comete cada método al tratar de predecir las primas obtenidas en la simulación de Montecarlo:")
print(medias)
