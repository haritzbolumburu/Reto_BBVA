import packages.Preprocesamiento as ppr
import statsmodels.api as stm
import pandas as pd
import os
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
from pandas.tseries.offsets import BDay, DateOffset, CustomBusinessDay
import seaborn as sns 


delay=ppr.retraso()

# Cargar ficheros 
print('Se van a cargar todos los ficheros originales.')
dataframes = ppr.read_data('./Datos/Originales')
print('Vamos a comprobar que se han cargado los ficheros:')
ppr.comprobar_lectura(dataframes)

# Análisis rápido
print('Una vez cargado los ficheros, vamos hacer un análisis rápido de las columnas e información de los dataframes')
ppr.analisis_datos(dataframes)

# Columna date como índice
print('Los ficheros deben someterse a algunos cambios para proceder a los modelos. Vamos a realizar unos primeros cambios')
print('Como estamos trabajando con datos de series temporales, colocamos la columna DATE como ÍNDICE')
for nombre_df, df in dataframes.items():
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)



# Adecuar todas las series a la misma franja horaria y longitud
print('Como se ve en el gráfico, no todas las series tienen la misma longitud, por lo que vamos a adecuarla. Además, como los datos son del mercado americano, restamos 6h a los datos.')
#PRIMERA FECHA EN COMÚN QUE TIENEN LOS DF
fechas_minimas = []
for nombre_df, df in dataframes.items():
    fechas_minimas.append(df.index.min())
fecha_minima = max(fechas_minimas)

#Quitar de los dataframes los datos anteriores a esa fecha
for nombre_df, df in dataframes.items():
    df.drop(df[df.index < fecha_minima].index, inplace=True)
fechas_maximas = []
for nombre_df, df in dataframes.items():
    fechas_maximas.append(df.index.max())
fecha_maxima = max(fechas_maximas)

# Quítale seis horas a todos los dataframes.
for nombre_df, df in dataframes.items():
    df.index = df.index - pd.Timedelta(hours=6)

# Creación de nuevas variables
print('Ahora se creará una nueva columna que dirá a qué día de la semana pertenece cada fecha para comprobar que solo tenemos datos de días hábiles.')
for nombre_df, df in dataframes.items():
        df['dayofweek'] = df.index.dayofweek
        df['work'] = df['dayofweek'].apply(lambda x: 0 if x > 4 else 1)

print('Como los fines de semana el mercado esta cerrado, vamos a comprobar que no hay datos para sábados ni domingos con un dataframe cualquiera (AAPL_marketstack)')
#print(AAPL_marketstack['dayofweek'].unique())
print('Solo hay fechas para 0:Lunes, 1:Martes, 2:Miércoles, 3:Jueves y 4:Viernes')

# Eliminar festivos
print('También eliminaremos fechas festivas')
print('Si es 1 de enero o 25 de diciembre, la columna work tendra valor de cero')
dates = ['2020-1-20', '2020-2-17', '2020-4-10', '2020-5-25', '2020-7-3', '2020-9-7', '2020-11-11', '2020-11-26', '2021-1-18', '2021-2-15', '2021-4-2', '2020-5-31', '2021-6-18', '2021-7-5', '2021-9-6', '2021-11-11', '2021-11-25']
ppr.eliminar_fechas_fes(dates,dataframes)

print('Ahora vamos a eliminar los registros para todos los días que tengan valor cero en la columna work')
ppr.eliminar_work(dataframes)

#quitar la columna date del index
for nombre_df, df in dataframes.items():
    df.reset_index(inplace=True)
    df.drop(columns=['work'], inplace=True)

#FRECUENCIA Y MISSINGS
print('Procedemos hacer el analisis de frecuencia y missings. Primero leemos los nuevos ficheros creados.')

#FRECUENCIA
print('Comprobamos las frecuencias muestrales')
print('Diaria')
ppr.frecuencia_diaria(dataframes)
print('Semanal')
ppr.frecuencia_semanal(dataframes)
print('Mensual')
ppr.frecuencia_mensual(dataframes)

#MISSINGS
print('Una vez analizada la frecuencia, vamos a observar y analizar los missings para un dataframe:')
for name,df in dataframes.items():
    variable=df['close']
    longitiud=len(df)
    print(f'{name}: missings en close: {variable.isna().sum()} ; {variable.isna().sum()/longitiud}')
#ppr.plot_missing_values_by_month(MMM_marketstack, date_column='date', value_column='close')

print('Corregimos y imputamos frecuencias')

for nombre_df, df in dataframes.items():
    df = df.sort_values(by='date')
    
print('imputación para poner valores de open en el siguiente día de close')
for nombre_df, df in dataframes.items():
    df=ppr.generar_close_open(df)
    dataframes[nombre_df] = df
    globals()[nombre_df]=df

print('Generar fechas completas. Formato = yyyy-mm-dd')
festivos = [pd.to_datetime('2020-06-19'), pd.to_datetime('2020-07-04'), pd.to_datetime('2020-12-25'),pd.to_datetime('2021-01-01'), pd.to_datetime('2021-06-19'), pd.to_datetime('2021-07-04'),pd.to_datetime('2020-01-20'), pd.to_datetime('2020-02-17'), pd.to_datetime('2020-04-10'),pd.to_datetime('2020-05-25'), pd.to_datetime('2020-07-03'), pd.to_datetime('2020-09-07'),pd.to_datetime('2020-11-11'), pd.to_datetime('2020-11-26'), pd.to_datetime('2021-01-18'),pd.to_datetime('2021-02-15'), pd.to_datetime('2021-04-02'), pd.to_datetime('2020-05-31'),pd.to_datetime('2021-06-18'), pd.to_datetime('2021-07-05'), pd.to_datetime('2021-09-06'),pd.to_datetime('2021-11-11'), pd.to_datetime('2021-11-25')]
dataframes=ppr.agregar_fechas_faltantes(dataframes, festivos)

print('Calcular todas las interpolaciones')
ppr.interpolacion_y_relleno(dataframes)



print('Seleccionamos interpolación cúbica y corregimos frecuencia')
print('DIARIA:')
dataframes_diarios=dataframes.copy()
for nombre_df, df in dataframes_diarios.items():
    df = df['close_cubic'].resample('D').median().to_frame()
    df=df.dropna()
    print(f'{nombre_df} -> len {len(df)}')
    print(f'--------- {nombre_df} primera fecha {df.index[0]}')
    print(f'--------- {nombre_df} ultima fecha {df.index[-1]}')
    dataframes_diarios[nombre_df] = df
    globals()[f'{nombre_df}_diario']=df


print('SEMANAL:')
dataframes_semanal=dataframes.copy()
for nombre_df, df in dataframes_semanal.items():
    df = df['close_cubic'].resample('W').median().to_frame()
    df=df.dropna()
    dataframes_semanal[nombre_df] = df
    globals()[f'{nombre_df}_semanal']=df


#CORRELACIONES
print('Correlaciones de los activos')
columnas=[i for i in dataframes.keys()]
df_cor=pd.DataFrame(columns=columnas)
for nombre_df, df in dataframes_diarios.items():
    df_cor[nombre_df]=df


ppr.obtener_correlaciones_extremas(df_cor.corr())
print(df_cor[['MRK_marketstack','GS_marketstack', 'JPM_marketstack', 'INTC_marketstack','TRV_marketstack','AXP_marketstack', 'CVX_marketstack', 'CAT_marketstack']].corr())


#guardar dataframes
ppr.guardar_dataframes_csv(dataframes_diarios,'./Datos/Transformados/diarios/')
ppr.guardar_dataframes_csv(dataframes_semanal,'./Datos/Transformados/semanales/')
print('Fin, datos guardados')