# LIBRERIAS

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.mstats import gmean
from numpy.linalg import cholesky
from scipy.stats import norm
import os
import random
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import packages.Modelado.funciones_modelado as mdl
from scipy.stats import norm
import pandas as pd
from scipy.stats import norm
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from tensorflow import keras

# SIMULACIONES

def premium_from_payoff(payoff, risk_free_rate, n_steps, n, t):
  """
  Devuelve la prima dado el payoff.


  Inputs:
  -------
  payoff: Valor de payoff.
  risk_free_rate: Tasa libre de riesgo.
  n_steps: Nº total de instantes.
  n: Instante temporal desde el que se simulan las distintas evoluciones.
  t: Valor de cada transicion temporal en años.


  Outputs:
  --------
  premium: Valor de la prima para la call.
  """

  return np.sum(np.mean(payoff,axis=0)*np.exp(-risk_free_rate*(n_steps-n)*t))



def payoff_y_prima(S_T, S_0, K, n, n_steps, risk_free_rate, num_assets, num_asian_dates, t = 0.25):

  """
  Inputs:
  -------
  S_T: Secuencia de valores de los activos.
  S_0: Vector de valores iniciales de los activos.
  K: Valor esperado de los activos.
  n: Instante temporal desde el que se simulan las distintas evoluciones.
  n_steps: Nº total de instantes.
  risk_free_rate: Tasa libre de riesgo.
  t: Valor de cada transicion temporal en años.


  Outputs:
  --------
  payoff: Valor de payoff para la cartera.
  prima: Valor de la prima para la call.
  """


  payoff = np.maximum(np.sum(np.prod(S_T/ S_0[None,:, None], axis=2)**(1/(num_assets * (num_asian_dates-1))), axis = 1)-K, 0)
  premium = premium_from_payoff(payoff=payoff, risk_free_rate=risk_free_rate, n_steps=n_steps, n=n, t=t)
  return payoff, premium


def MonteCarlo(initial_maturity, S_0, K, num_sims, num_assets, num_asian_dates, value_date_index, correl_matrix,risk_free_rate, vols):
    '''
    Calcula un determinado nº de simulaciones de la evolución del valor de los subyacentes, el payoff correspondiente a la cartera y el valor de la prima.

    Inputs:
    -------
    * initial_maturity (float): maturity of the product as seen on initial spot fixing date. (Years)
    * S_0 (float): initial spot prices of the assets
    * num_asian_dates (int): number of asian dates. Notice that this includes the initial date, which fixes the initial spot values.
    * value_date_index (int): index, within the array of asian dates indicating the value date. (Actual step)
    * risk_free_rate (float): risk free rate
    * num_assets (int): number of underlying assets.
    * vols (array(float)): array of indiv assets vols.
    * correl_matrix (array(float, float)): matrix of correlations
    * nim_sims (int): number of simulations


    Outputs:
    --------
    returns: Simulaciones del valor del subyacente.
    payoff: Valor del payoff para cada simulación.
    prima: Valor de la prima de la call.
    '''

    # Inputs:

    # Simulation of assets up to value date:

    init_time_array = np.linspace(0, initial_maturity, num_asian_dates)

    delta_t = initial_maturity / (num_asian_dates - 1)
    num_steps = value_date_index
    num_remaining_steps = num_asian_dates - value_date_index -1

    # Independent brownians
    inc_W = np.random.normal(size=(num_assets, num_steps), scale = np.sqrt(delta_t))

    # Cholesky matrix
    m = cholesky(correl_matrix)

    # Correlated brownians
    inc_W_correl = np.matmul(m, inc_W)

    # Independent brownian motion (the axis order is done to be able to correlate them with a matrix multiplication)
    inc_W_remaining = np.random.normal(size=(num_sims, num_remaining_steps, num_assets), scale = np.sqrt(delta_t))

    # We correlate them
    inc_W_correl_remaining = np.matmul(inc_W_remaining, m.T)

    # We transpose the 3D matrix of correlated B. motion (path, asset, time step)

    inc_W_correl_remaining = inc_W_correl_remaining.transpose([0,2,1])

    aux = np.repeat(inc_W_correl[None,...],num_sims,axis=0)

    # We attach the brownians obtained from t= 0 to value date

    inc_W_correl_total = np.concatenate((aux, inc_W_correl_remaining), axis = 2)

    # We compute exponential returns

    gross_rets_total = np.exp((risk_free_rate - 0.5 *vols **2) * delta_t + vols * inc_W_correl_total)

    # We simulate the underlyings

    S_T = np.cumprod(np.concatenate((np.repeat(S_0.reshape(-1,1)[None,...],num_sims,axis=0), gross_rets_total), axis = 2), axis = 2)

    # We compute the returns

    rets = S_T[:,:,1:] / np.repeat(S_0.reshape(-1,1)[None,...],num_sims,axis=0)
    
    payoff, premium = payoff_y_prima(S_T, S_0, K=K, n_steps=20, n=0, num_assets=num_assets, num_asian_dates=num_asian_dates, risk_free_rate=risk_free_rate)


    rets = rets.transpose([0,2,1])

    return rets, payoff, premium



# FORMULA CERRADA

def Black(Forward, Strike, TTM, rate, Vol, IsCall,N):

  '''
  Inputs:
  -------
    Forward (float): Forward value
    Strike (float): strike price
    TTM (float): time to maturity in years
    rate (float): risk free rate
    Vol (float): volatility
    IsCall (bool): True if call option, False if put option

  Outputs:
  --------
    Option premium (float)
  '''

  if TTM >0:

    d1 = (np.log(Forward/Strike) + (Vol*Vol/2)*TTM)/(Vol*np.sqrt(TTM))
    d2 = (np.log(Forward/Strike) + (- Vol*Vol/2)*TTM)/(Vol*np.sqrt(TTM))

    if IsCall:

      return (Forward*N(d1)-Strike*N(d2))*np.exp(-rate*TTM)

    else:

      return (-Forward*N(-d1)+Strike*N(-d2))*np.exp(-rate*TTM)

  else:

    if IsCall:

      return np.maximum(Forward-Strike,0)

    else:

      return np.maximum(-Forward+Strike,0)
    

def BasketGeomAsian(num_asian_dates, value_date_index, risk_free_rate, num_assets, assets_vol, assets_correl, initial_maturity, price_history, IsCall,N):
  '''
  Inputs:
  -------
  * num_asian_dates (int): number of asian dates. Notice that this includes the initial date, which fixes the initial spot values.
  * value_date_index (int): index, within the array of asian dates indicating the value date.
  * risk_free_rate (float): risk free rate
  * num_assets (int): number of underlying assets.
  * assets_vol (array(float)): array of indiv assets vols.
  * assets_correl (array(float, float)): matrix of correlations
  * initial_maturity (float): maturity of the product as seen on initial spot fixing date.
  * price_history (array(float, float)): history of fixings of the underlyings up to value date. Assets per row, time steps per column. Dimensions as follows: (n_simulations, asets, time_steps)
  * IsCall (bool): True if call option, False if put option
  Outputs:
  --------
  * Option price (float)
  '''

  init_time_array = np.linspace(0, initial_maturity, num_asian_dates)

  pending_times_array = init_time_array[value_date_index+1:] - init_time_array[value_date_index]

  mu = np.sum(risk_free_rate - 0.5*assets_vol*assets_vol)*np.sum(pending_times_array) / (num_assets * (num_asian_dates-1))

  diag_vol = np.diag(assets_vol.reshape(-1))

  cov_matrix = np.matmul(diag_vol, np.matmul(assets_correl, diag_vol))

  xx, yy = np.meshgrid(pending_times_array, pending_times_array, sparse=True)
  z = np.minimum(xx, yy)

  V = np.sum(cov_matrix) * np.sum(z) / (num_assets*num_assets*(num_asian_dates-1)*(num_asian_dates-1))

  Forward = np.power(np.prod(price_history[:, 1:value_date_index+1] / price_history[:,0].reshape(-1,1)),1.0/(num_assets * (num_asian_dates-1)))

  Forward *= np.power(np.prod(price_history[:,value_date_index] / price_history[:,0]), (num_asian_dates-value_date_index-1)/(num_assets * (num_asian_dates-1)))

  Forward *= np.exp(mu + 0.5 * V)

  remaining_maturity = initial_maturity - init_time_array[value_date_index]


  return Black(Forward, 1.0, remaining_maturity, risk_free_rate,np.sqrt(V / remaining_maturity), IsCall,N)



# FUNCIONES CREADAS


def plot_history(history,title, plot_list=[]):
  """Grafica la evolucion de las metricas segun avanzan las epocas de la epoca de entrenamiento

  Args:
      history (keras.callbacks.History):Registro obtenido del entrenamiento del modelo
      title (str): Titulo del grafico
      plot_list (list, optional): Lista de metricas a graficar. Defaults to [].
  """
  fig = plt.figure(figsize=(10, 10))
  plt.xlabel("Epoch")
    
  colors = ['#225380', '#2DC7C9']
    
  for i, plot in enumerate(plot_list):
        color = colors[i]
        plt.plot(history.epoch, history.history[plot], label=plot, color=color)
  plt.title(title)
  plt.legend()
  plt.rc('font', size=15)
  plt.show()

def model_test(ytrue,ypred):
  """Calucla el Mean Square Error(MAE) y Mean Absolute Error(MAE) de los valores predichos en cuanto a los reales

  Args:
      ytrue (list): valores reales
      ypred (list): valores predichos

  Returns:
      float: MSE y MAE
  """
  mse = mean_squared_error(ytrue, ypred)
  mae=mean_absolute_error(ytrue,ypred)

  return mse, mae


def train_val_loss(history,loss_list,metric_list):
  """Ejecuta la funcion 'plot_histoy'

  Args:
      history (keras.callbacks.History): Registro obtenido del entrenamiento del modelo
      loss_list (list): lista con la metrica empleada como loss function
      metric_list (list): lista con otra metrica empleada en el entrenamiento
  """
  mdl.plot_history(history,loss_list)
  mdl.plot_history(history,metric_list)


def plot_error_distribution(ytrue, ypred):
  """Grafica el error que comete el modelo en la prediccion de la prima

  Args:
      ytrue (list): Valores reales
      ypred (list): Valores predichos

  Raises:
      ValueError: Longitudes de ytru  e ypred son diferentes

  Returns:
      plotly.graph_objs._figure.Figure: Grafico de distribucion de los errores
  """
  if len(ytrue) != len(ypred):
        raise ValueError("Las longitudes de ytrue e ypred deben ser iguales.")

  errors = ytrue - ypred.flatten()

    # Crear un DataFrame para Plotly
  df = pd.DataFrame({'Errores': errors, 'Numero de predicciones': range(1, len(errors)+1)})

    # Gráfico interactivo con Plotly Express (Gráfico de dispersión)
  fig = px.scatter(df, x='Numero de predicciones', y='Errores', title='Distribución de Errores Medios Absolutos')

    # Personalizar el color del marcador
  fig.update_traces(marker=dict(color='#225380'))

  fig.update_layout(showlegend=True, plot_bgcolor='white', font=dict(size=15))

  return fig

def predicciones_finales(model,rets,initial_maturity,S_0,num_sims,num_assets,num_asian_dates,correl_matrix,risk_free_rate,vols,rho,TTM,N):
  """_summary_

  Args:
      model (keras.Sequential): Modelo Red Recurrente
      rets (array): Valores de los precios en los pasos de las simulaciones creadas
      initial_maturity (float): maturity of the product as seen on initial spot fixing date. (Years)
      S_0 (float): initial spot prices of the assets
      num_sims (int): number of simulations
      num_assets (int): number of underlying assets.
      num_asian_dates (int): number of asian dates. Notice that this includes the initial date, which fixes the initial spot values.
      correl_matrix (array(float, float)): matrix of correlations
      risk_free_rate (float): risk free rate
      vols (array(float)): array of indiv assets vols.
      rho (list): matrix of correlations
      TTM (int): initial maturity
      N (method): norm.cdf

  Returns:
      list: lista con las primas y el numero de simulaciones y pasos empleados para predecirlas
  """
  rets2 = rets.transpose([0,2,1])
  n_sim = 0

  df_comp = []
  prima_montecarlo=[]
  prima_modelo=[]
  prima_bbva=[]
  for n_sim in range(100):
            for long in range(1,20):
                    _, _, premium = mdl.MonteCarlo(initial_maturity, S_0, K=3, num_sims=num_sims, num_assets=num_assets, num_asian_dates=num_asian_dates, value_date_index=long, correl_matrix=correl_matrix, risk_free_rate=risk_free_rate, vols=vols)
                    prima_montecarlo.append(premium)
                    bbva=mdl.BasketGeomAsian(num_asian_dates,long-1,risk_free_rate,num_assets, vols, rho,initial_maturity, rets2[n_sim,:,:long],True,N)
                    prima_bbva.append(bbva)
                    prediccion=model.predict(np.expand_dims(rets[n_sim,:long,:],0), verbose=0)
                    prediccion_presente=mdl.premium_from_payoff(np.expand_dims(prediccion[0][0],0),risk_free_rate,20,long,TTM)
                    prima_modelo.append(prediccion_presente)

                    df_comp.append([n_sim, long, premium,bbva,prediccion[0][0]])
  return df_comp

def calculos_errores(df_comp):
  """Calcula el Mean Square Error(MSE) y el Mean Absolute Error(MAE) de las predicciones de las primas

  Args:
      df_comp (list): lista con las primas obtenidas y el numero de simulaciones y pasos

  Returns:
      pd.DataFrame: Data frame con los errores calculados
  """
  resultados=pd.DataFrame(df_comp, columns=["n_sim", "long", "premium","bbva","prediccion"])
  resultados['MSE_RED_RECURRENTE']=(resultados['premium']-resultados['prediccion'])**2
  resultados['MAE_RED_RECURRENTE']=np.abs(resultados['premium']-resultados['prediccion'])
  resultados['MSE_BBVA']=(resultados['premium']-resultados['bbva'])**2
  resultados['MAE_BBVA']=np.abs(resultados['premium']-resultados['bbva'])

  return resultados

def grafico__bbva_red(df, titulo):
  """Grafica los errores segun avance el valor del indice del dataframe

  Args:
      df (pd.DataFrame): data frame que contiene los errores al predecir las primas estableciendo como indice el eje x
      titulo (str): Titulo del grafico
  """
  fig = make_subplots(rows=2, cols=1, subplot_titles=['Error Absoluto Medio', 'Error Cuadrático Medio'])
  colors = ['#225380', '#2DC7C9']
    # Agregar gráfico de línea para Columnas 1 y 2
  fig.add_trace(go.Line(x=df.index, y=df['MAE_BBVA'], mode='lines', name='BBVA', line_color=colors[0]), row=1, col=1)
  fig.add_trace(go.Line(x=df.index, y=df['MAE_RED_RECURRENTE'], mode='lines', name='Red Recurrente',line_color=colors[1]), row=1, col=1)

    # Agregar gráfico de línea para Columnas 3 y 4
  fig.add_trace(go.Line(x=df.index, y=df['MSE_BBVA'], mode='lines', name='BBVA',line_color=colors[0],showlegend=False), row=2, col=1)
  fig.add_trace(go.Line(x=df.index, y=df['MSE_RED_RECURRENTE'], mode='lines', name='Red Recurrente',line_color=colors[1],showlegend=False), row=2, col=1)

    # Actualizar diseño y mostrar la figura
  fig.update_layout(title_text=titulo, showlegend=True, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',font=dict(size=15))
  fig.update_yaxes(title_text='Error', row=1, col=1)
  fig.update_yaxes(title_text='Error', row=2, col=1)
  fig.show()

def grafico_comparacion_final(df):
  """Realiza un grafico de barras a partir de un dataframe

  Args:
      df (pandas.Series): Valores de un dataset agrupado
  """
  colores = {'MAE_BBVA': '#225380', 'MAE_RED_RECURRENTE': '#2DC7C9', 'MSE_BBVA': '#225380', 'MSE_RED_RECURRENTE': '#2DC7C9'}

    # Crear un gráfico de barras con colores personalizados y sin fondo
  fig = px.bar(df, x=df.index, y=df.values, color=df.index, color_discrete_map=colores, labels={'x': 'Métrica', 'y': 'Valor'})

    # Configurar el diseño sin fondo y mostrar el gráfico
  fig.update_layout(title_text='Errores MAE y MSE para BBVA y Red Recurrente', xaxis_title='Métrica', yaxis_title='Valor', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',font=dict(size=15))
  fig.update_layout(showlegend=False)
  fig.show()

def get_model():
  """Crea la estructura del modelo

  Returns:
      keras.Sequential: Red recurrente
  """
  model = keras.models.Sequential([
  keras.layers.GRU(20, activation='tanh', input_shape=(None, 3), return_sequences=True),
  keras.layers.Dense(20, activation='linear', name='dense_1'),
  keras.layers.GRU(20, activation='tanh', return_sequences=True),
  keras.layers.Dense(10, activation='linear', name='dense_2'),
  keras.layers.GRU(10, activation='tanh'),
  keras.layers.Dense(1, activation='linear', name='output')
    ])
  return model
   
def train_model(model,X_train,Y_train,X_val,Y_val ):
  """_summary_

  Args:
      model (keras.Sequential): Red recurrente
      X_train (array): datos de entrenamiento 
      Y_train (array): target del entrenamiento
      X_val (array): datos de validacion
      Y_val (array): target de validacion

  Returns:
      keras.Sequential: Red Recurrente entrenada
  """
  model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics='mae')
  earlystopping=keras.callbacks.EarlyStopping(patience=15,min_delta=0.00001)
  history = model.fit(
      x=X_train, 
      y=Y_train, 
      batch_size=40, 
      epochs=50,
      callbacks=[keras.callbacks.ModelCheckpoint(filepath='Models_checkpoint/model.hdf5'),earlystopping],
      validation_data=(X_val, Y_val)
  )

  return model