import packages.Preprocesamiento as ppr
import statsmodels.api as stm
import pandas as pd
import os
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
from pandas.tseries.offsets import BDay, DateOffset, CustomBusinessDay


def retraso()->int:
    """pide al usuario el numero de segundos que quiere que pase entre comando y comando
    Returns:
        int: EL numero de segundos que pasaran entre comando y comando
    """
    retraso=float(input('Especifica el retraso entre codigos.(En segundos):'))
    print(f'El retraso seleccionado han sido {retraso} segundos.')
    return retraso

def read_data(carpeta: str):
    """Lectura de un fichero csv del directorio Datos/Originales

    Args:
        filename (str): Nombre del fichero

    Returns:
        pd.DataFrame: El fichero en formato Dataframe
    """
    archivos_csv = [archivo for archivo in os.listdir(carpeta) if archivo.endswith('.csv')]
    dataframes = {}
    for archivo in archivos_csv:
        ruta_completa = os.path.join(carpeta, archivo)
        nombre_df = os.path.splitext(archivo)[0] 
        df = pd.read_csv(ruta_completa)
        dataframes[nombre_df] = df
    return dataframes

def comprobar_lectura(dataframes):
    for nombre_df, df in dataframes.items():
        globals()[nombre_df] = df
        print('Fichero cargado:' + nombre_df)

def eliminar_fechas_fes(dates, dataframes):
    for nombre_df, df in dataframes.items():
        for date in dates:
            date = pd.to_datetime(date)
            ppr.festivos(df, date.year, date.month, date.day)

def analisis_datos(dataframes):
    """
    Realiza un análisis básico de datos para cada DataFrame en un diccionario de dataframes.

    Args:
        dataframes (dict): Un diccionario que contiene dataframes de pandas.

    Returns:
        None
    """
    for nombre_df, df in dataframes.items():
        print(f'Análisis para el DataFrame: {nombre_df}')
        print(f'Las columnas que tiene este dataframe son: {list(df.columns)}')
        print('-----------------------------------------------------------------')
        print('-----------------------------------------------------------------')
        print('Aquí se muestra alguna información extra:')
        print(df.info())
        print('-----------------------------------------------------------------')
        print('-----------------------------------------------------------------')
        df_missings = df.isna().sum().reset_index(name='missings')
        suma_missings = int(df_missings['missings'].sum())
        print(f'Hay un total de {suma_missings} missings en todo el dataframe')
        print('\n\n')



def festivos(df, año, mes, dia):
    """
    Marca un día específico como festivo en un DataFrame.
    
    Parámetros:
    df (pandas.DataFrame): El DataFrame en el que se marcará el día como festivo.
    año (int): Año del festivo.
    mes (int): Mes del festivo (1 para enero, 2 para febrero, etc.).
    dia (int): Día del festivo.
    
    Returns:
    None
    """
    df.loc[(df.index.year == año) & (df.index.month == mes) & (df.index.day == dia), 'work'] = 0
    
def eliminar_work(dataframes):
    for nombre_df, df in dataframes.items():
        df.drop(df[df['work'] == 0].index, inplace=True)

def guardar_csv(dataframes, directorio):
    """
    Elimina la columna 'work' de cada dataframe del diccionario y guarda los dataframes modificados como archivos CSV.

    Args:
        dataframes (dict): Un diccionario que contiene dataframes de pandas.
        directorio (str): La ruta del directorio donde se guardarán los archivos CSV.

    Returns:
        None
    """
    # Verifica si el directorio existe, si no, créalo
    if not os.path.exists(directorio):
        os.makedirs(directorio)

    for nombre_df, df in dataframes.items():
        df.drop(columns=['work'], inplace=True)
        nombre_csv = nombre_df + '.csv'
        df.to_csv('./Datos/Transformados/' + nombre_csv)

def obtener_precio_apertura(group):
    """
    Obtiene el precio de apertura del primer registro en un grupo de datos de un DataFrame.
    
    Parámetros:
    group (pandas.DataFrame): Un grupo de datos (subconjunto del DataFrame principal) sobre el cual se calculará el precio de apertura.
    
    Returns:
    float: El precio de apertura del primer registro en el grupo de datos.
    """
    return group.iloc[0]['open']

def obtener_precio_cierre(group):
    """
    Obtiene el precio de cierre del último registro en un grupo de datos de un DataFrame.
    
    Parámetros:
    group (pandas.DataFrame): Un grupo de datos (subconjunto del DataFrame principal) sobre el cual se calculará el precio de cierre.
    
    Returns:
    float: El precio de cierre del último registro en el grupo de datos.
    """
    return group.iloc[-1]['close']


def frecuencia_diaria(dataframes):
    """
    Calcula la frecuencia diaria de los datos en los dataframes proporcionados, excluyendo fines de semana y festivos.
    
    Parámetros:
    dataframes (dict): Un diccionario donde las claves son los nombres de los dataframes y los valores son los dataframes.
    
    Returns:
    None
    """
    # Define un rango de fechas desde 2017 hasta 2022
    fecha_inicio = pd.to_datetime('2020-06-02')
    fecha_fin = pd.to_datetime('2021-07-21')

    # Define una lista de festivos (puedes personalizarla según tu país o región)
    festivos = [pd.to_datetime('2020-06-19'),pd.to_datetime('2020-07-04'),pd.to_datetime('2020-12-25'), pd.to_datetime('2021-01-01'), pd.to_datetime('2021-06-19'),
                pd.to_datetime('2021-07-04'), pd.to_datetime('2020-01-20'), pd.to_datetime('2020-02-17'),pd.to_datetime('2020-04-10'),pd.to_datetime('2020-05-25'),
            pd.to_datetime('2020-07-03'),pd.to_datetime('2020-09-07'),pd.to_datetime('2020-11-11'),pd.to_datetime('2020-11-26'),       
            pd.to_datetime('2021-01-18'),pd.to_datetime('2021-02-15'),pd.to_datetime('2021-04-02'),pd.to_datetime('2020-05-31'),
            pd.to_datetime('2021-06-18'),pd.to_datetime('2021-07-05'),pd.to_datetime('2021-09-06'),pd.to_datetime('2021-11-11'),
            pd.to_datetime('2021-11-25')]

    # Define un CustomBusinessDay para excluir los fines de semana (sábado y domingo)
    cbd = CustomBusinessDay(weekmask='Mon Tue Wed Thu Fri')

    # Crea una serie de fechas que no sean festivos ni fines de semana
    fechas_no_weekends = pd.date_range(start=fecha_inicio, end=fecha_fin, freq=cbd)
    fechas_no_weekends_no_festivos = fechas_no_weekends[~fechas_no_weekends.isin(festivos)]
    for name,df in dataframes.items():
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values(by='date', inplace=True)
        df['solo_fecha']=df['date'].dt.date
        # Utiliza np.unique() para obtener los valores únicos y sus conteos
        valores_unicos, conteos = np.unique(fechas_no_weekends_no_festivos.isin(df['solo_fecha']), return_counts=True)

        # Crea un diccionario que muestre los valores únicos y sus conteos
        diccionario_conteo = dict(zip(valores_unicos, conteos))

        # Imprime el resultado
        print(f"{name}: {diccionario_conteo}")

def frecuencia_semanal(dataframes):
    """
    Calcula la frecuencia semanal de los datos en los dataframes proporcionados y muestra la cantidad de semanas 
    con y sin datos para cada dataframe en el período desde '2020-06-02' hasta '2021-07-21'.
    
    Parámetros:
    dataframes (dict): Un diccionario donde las claves son los nombres de los dataframes y los valores son los dataframes.
    
    Returns:
    None
    """
    # Crear una lista de fechas desde '2020-06-02' hasta '2021-07-21' con información de la semana
    fechas_con_semana = pd.date_range(start='2020-06-02', end='2021-07-21', freq='W-MON')

    # Convertir las fechas al formato deseado ('%Y/%U') y almacenarlas en una lista
    lista_anio_semana = [fecha.strftime('%Y/%U') for fecha in fechas_con_semana]

    # Crear un diccionario para almacenar los resultados por nombre
    resultados_por_nombre = {}

    for name, df in dataframes.items():
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values(by='date', inplace=True)
        df['solo_fecha'] = df['date'].dt.date
        df['semana'] = df['date'].dt.strftime('%Y/%U')  # Obtener el año y la semana en formato 'año/semana'

        # Filtrar el DataFrame para incluir solo las semanas en lista_anio_semana
        df_filtrado = df[df['semana'].isin(lista_anio_semana)]

        # Crear un DataFrame que contenga True si hay al menos un dato en esa semana, False de lo contrario
        semanas_con_datos = df_filtrado.groupby('semana')['date'].count() > 0

        # Contar la cantidad de True (semanas con datos) y False (semanas sin datos)
        conteo_semanas = semanas_con_datos.value_counts().to_dict()

        # Almacenar el resultado en el diccionario por nombre
        resultados_por_nombre[name] = conteo_semanas

    # Imprimir los resultados
    for name, resultados in resultados_por_nombre.items():
        print(f"{name}: {resultados}")

def frecuencia_mensual(dataframes):
    """
    Calcula la frecuencia mensual de los datos en los dataframes proporcionados y verifica la presencia de todos
    los meses desde '2020-06' hasta '2021-07' en los datos de cada dataframe.
    
    Parámetros:
    dataframes (dict): Un diccionario donde las claves son los nombres de los dataframes y los valores son los dataframes.
    
    Returns:
    None
    """
    # PREPARACION DE LA LISTA DE FECHAS

    # Crear una lista de fechas desde '2010/03' hasta '2018/05'
    fechas = pd.date_range(start='2020-06', end='2021-07', freq='MS')

    # Convertir las fechas al formato deseado ('%Y/%m') y almacenarlas en una lista
    lista_anio_mes = [fecha.strftime('%Y/%m') for fecha in fechas]

    # Crear una lista de fechas desde '2010/03' hasta '2018/05'
    fechas = pd.date_range(start='2020-06', end='2021-07', freq='MS')

    # Convertir las fechas al formato deseado ('%Y/%m') y almacenarlas en una lista
    lista_anio_mes = [fecha.strftime('%Y/%m') for fecha in fechas]
    lista_anio_mes = set(lista_anio_mes)
    print(f'tiene que haber {len(lista_anio_mes)} meses')


    for name,df in dataframes.items():
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values(by='date', inplace=True)
        df['anio_mes'] = df['date'].dt.strftime('%Y/%m')
        meses_presentes = set(df['anio_mes'])
        print(len(meses_presentes))
        # # Utiliza np.unique() para obtener los valores únicos y sus conteos
        # valores_unicos, conteos = np.unique(lista_anio_mes.isin(df['anio_mes']), return_counts=True)

        # # Crea un diccionario que muestre los valores únicos y sus conteos
        # diccionario_conteo = dict(zip(valores_unicos, conteos))

        # # Imprime el resultado
        # print(f"{name}: {diccionario_conteo}")
        todos_los_meses_presentes = lista_anio_mes.issubset(meses_presentes)
        print(f'{name}: {todos_los_meses_presentes}')
def plot_missing_values_by_month(dataframe, date_column='date', value_column='close'):
    """
    Visualiza el número de instancias con valores ausentes por mes.

    Args:
        dataframe (pd.DataFrame): El dataframe que contiene las columnas 'date' y 'close' u otras columnas relevantes.
        date_column (str): El nombre de la columna que representa las fechas en el dataframe. Por defecto, 'date'.
        value_column (str): El nombre de la columna que contiene los valores para los cuales se contabilizan los valores ausentes. Por defecto, 'close'.
    """
    miss_month = dataframe.set_index(date_column)[value_column].resample('M').apply(lambda x: x.isnull().sum())
    plt.figure(figsize=(10, 6))
    miss_month.plot(kind='bar', color='skyblue')
    plt.title('Número de Instancias con Valores Ausentes por Mes')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def generar_close_open(dataframe):
    """
    Genera una nueva columna 'close_open' en el dataframe, llenando los valores faltantes en 'close' 
    con el valor de la siguiente instancia de 'open'.

    Args:
        dataframe (pd.DataFrame): El dataframe original que contiene las columnas 'open' y 'close'.

    Returns:
        pd.DataFrame: Un nuevo dataframe que incluye la columna 'close_open' con los valores faltantes llenados.
    """
    # Copia el dataframe para evitar modificar el original
    df = dataframe.copy()
    
    # Llena los valores faltantes en 'close' con el valor de la siguiente instancia de 'open'
    df['close_open'] = df['close'].fillna(df['open'].shift(1))
    
    return df

def agregar_fechas_faltantes(dataframes, festivos):
    """
    Agrega las fechas faltantes a los dataframes según un rango de fechas y excluyendo días festivos.

    Args:
        dataframes (dict): Un diccionario que contiene los dataframes con columnas 'date' y otras.
        festivos (list): Una lista de fechas festivas.

    Returns:
        dict: Un diccionario que contiene los dataframes actualizados con las fechas faltantes agregadas.
    """
    cbd = CustomBusinessDay(weekmask='Mon Tue Wed Thu Fri')
    fechas_completas = pd.date_range(start='2020-06-02', end='2021-06-30', freq=cbd)
    fechas_completas = set(fechas_completas.date) - set(festivos)

    for nombre_df, df in dataframes.items():
        df['date'] = pd.to_datetime(df['date'])
        df['date_date'] = df['date'].dt.date
        fechas_faltantes = set(fechas_completas) - set(df['date_date'])
        nuevos_registros = pd.DataFrame({'date_date': list(fechas_faltantes)})
        nuevos_registros['date'] = pd.to_datetime(nuevos_registros['date_date'])
        nuevos_registros['date'] = nuevos_registros['date'].dt.strftime('%Y-%m-%d %H:%M:%S+00:00')
        df = pd.concat([df, nuevos_registros], ignore_index=True)
        df = df.sort_values(by='date_date')
        df.set_index('date', inplace=True)
        df.index = pd.to_datetime(df.index, utc=True)
        dataframes[nombre_df] = df

    return dataframes

def interpolacion_y_relleno(dataframes):
    """
    Aplica diferentes métodos de interpolación y relleno para las columnas 'close' en los dataframes.

    Args:
        dataframes (dict): Un diccionario que contiene los dataframes con una columna 'close'.

    Returns:
        dict: Un diccionario que contiene los dataframes actualizados con las nuevas columnas interpoladas y rellenadas.
    """
    for nombre_df, df in dataframes.items():
        # Interpolación cúbica
        df['close_cubic'] = df['close'].interpolate(method='cubic')
        
        # Interpolación lineal
        df['close_linear'] = df['close'].interpolate(method='linear')
        
        # Relleno hacia adelante (forward fill)
        df['last_close'] = df['close'].fillna(method='ffill')
        
        # Relleno hacia atrás (backward fill)
        df['next_close'] = df['close'].fillna(method='bfill')
        
        dataframes[nombre_df] = df
        globals()[nombre_df] = df

# def plot_interpolations(dataframe, start_date='2021-02-01'):
#     """
#     Grafica las interpolaciones y otras columnas de un dataframe a partir de una fecha específica.

#     Args:
#         dataframe (pd.DataFrame): El dataframe que contiene las columnas a graficar.
#         start_date (str): La fecha a partir de la cual se realizarán las gráficas. Por defecto, '2021-02-01'.
#     """
#     plt.figure(figsize=(15, 15))

#     # Primer gráfico - Interpolación cúbica
#     plt.subplot(5, 1, 1)
#     plt.plot(dataframe[dataframe.index > start_date]['close_cubic'].astype(float), color='red')
#     plt.plot(dataframe[dataframe.index > start_date]['close'].astype(float), color='blue')
#     plt.title('Interpolación cúbica')

#     # Segundo gráfico - Interpolación lineal
#     plt.subplot(5, 1, 2)
#     plt.plot(dataframe[dataframe.index > start_date]['close_linear'].astype(float), color='red')
#     plt.plot(dataframe[dataframe.index > start_date]['close'].astype(float), color='blue')
#     plt.title('Interpolación lineal')

#     # Tercer gráfico - Valores de open en close
#     plt.subplot(5, 1, 3)
#     plt.plot(dataframe[dataframe.index > start_date]['close_open'].astype(float), color='red')
#     plt.plot(dataframe[dataframe.index > start_date]['close'].astype(float), color='blue')
#     plt.title('Valores de open en close')

#     # Cuarto gráfico - Last Observation Carried Forward
#     plt.subplot(5, 1, 4)
#     plt.plot(dataframe[dataframe.index > start_date]['last_close'].astype(float), color='red')
#     plt.plot(dataframe[dataframe.index > start_date]['close'].astype(float), color='blue')
#     plt.title('Last Observation Carried Forward')

#     # Quinto gráfico - Next Observation Carried Backward
#     plt.subplot(5, 1, 5)
#     plt.plot(dataframe[dataframe.index > start_date]['next_close'].astype(float), color='red')
#     plt.plot(dataframe[dataframe.index > start_date]['close'].astype(float), color='blue')
#     plt.title('Next Observation Carried Backward')

#     plt.tight_layout()
#     plt.show()


# def comparar_interpolaciones(dataframe, fecha_inicio='2021-02-01'):
#     """
#     Compara las diferentes interpolaciones y otras columnas de un dataframe a partir de una fecha específica.

#     Args:
#         dataframe (pd.DataFrame): El dataframe que contiene las columnas a comparar.
#         fecha_inicio (str): La fecha a partir de la cual se realizará la comparación. Por defecto, '2021-02-01'.
#     """
#     plt.plot(dataframe[dataframe.index > fecha_inicio]['close_cubic'], color='red', label='cubic')
#     plt.plot(dataframe[dataframe.index > fecha_inicio]['close_linear'], color='green', label='linear')
#     plt.plot(dataframe[dataframe.index > fecha_inicio]['close'], color='blue', label='open_day')
#     plt.plot(dataframe[dataframe.index > fecha_inicio]['last_close'], color='orange', label='last')
#     plt.plot(dataframe[dataframe.index > fecha_inicio]['next_close'], color='grey', label='next')
#     plt.legend()
#     plt.title('Comparación de interpolaciones')
#     plt.show()

def drawdown(precios, tipo=1):
    """
    Calcula y grafica el drawdown diario y el máximo drawdown diario en un periodo de tiempo.

    Args:
        precios (pd.Series or pd.DataFrame): Precios de un activo financiero a lo largo del tiempo.
        tipo (int): Tipo de cálculo de drawdown.
                    1: Drawdown diario basado en rendimientos porcentuales (predeterminado).
                    2: Drawdown diario basado en precios.
                    Otro valor: Drawdown diario basado en rendimientos porcentuales.

    Returns:
        pd.Series, pd.Series: Una serie que representa el drawdown diario y otra serie que representa el 
                             máximo drawdown diario en un periodo de tiempo.
    """
    if tipo == 1:
        rtb = precios.pct_change().dropna()
        roll_max = rtb.rolling(center=False, min_periods=1, window=252).max()
        daily_draw_down = rtb / roll_max - 1
        max_daily_draw_down = daily_draw_down.rolling(center=False, min_periods=1, window=252).min()
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(rtb.index, daily_draw_down, label="Drawdown diario")
        ax.plot(rtb.index, max_daily_draw_down, label="Máximo drawdown diario en ventana de tiempo")
        ax.legend()
        plt.show()
        return daily_draw_down, max_daily_draw_down
    elif tipo == 2:
        roll_max = precios.rolling(center=False, min_periods=1, window=252).max()
        daily_draw_down = precios / roll_max - 1
        max_daily_draw_down = daily_draw_down.rolling(center=False, min_periods=1, window=252).min()
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(precios.index, daily_draw_down, label="Drawdown diario")
        ax.plot(precios.index, max_daily_draw_down, label="Máximo drawdown diario en ventana de tiempo")
        ax.set_ylim(-0.5, 0.2)
        ax.legend()
        plt.show()
        return daily_draw_down, max_daily_draw_down
    else:
        rtb = precios.pct_change().dropna()
        roll_max = rtb.rolling(center=False, min_periods=1, window=252).max()
        daily_draw_down = rtb / roll_max - 1
        max_daily_draw_down = daily_draw_down.rolling(center=False, min_periods=1, window=252).min()
        return daily_draw_down, max_daily_draw_down

def obtener_correlaciones_extremas(matriz_correlacion):
    """
    Encuentra las correlaciones más altas positivas y las correlaciones más bajas negativas en una matriz de correlación.

    Args:
        matriz_correlacion (pd.DataFrame): Una matriz de correlación de activos financieros.

    Returns:
        list, list: Una lista de tuplas que contiene las correlaciones positivas más altas y otra lista de
                   tuplas que contiene las correlaciones negativas más bajas.
    """
    # Obtener las correlaciones positivas más altas
    correlaciones_positivas = matriz_correlacion.unstack().sort_values(ascending=False)
    top_positivas = correlaciones_positivas[correlaciones_positivas < 1].head(10)

    # Obtener las correlaciones negativas más bajas
    correlaciones_negativas = matriz_correlacion.unstack().sort_values()
    top_negativas = correlaciones_negativas[correlaciones_negativas > -1].head(10)

    # Crear listas para almacenar los resultados
    resultados_positivos = []
    resultados_negativos = []

    # Iterar sobre las correlaciones positivas
    for index, valor in top_positivas.items():
        acciones = index
        correlacion = valor
        resultados_positivos.append((acciones, correlacion))

    # Iterar sobre las correlaciones negativas
    for index, valor in top_negativas.items():
        acciones = index
        correlacion = valor
        resultados_negativos.append((acciones, correlacion))

    return resultados_positivos, resultados_negativos

def calcular_metricas_financieras(dataframes_diarios):
    """
    Calcula las métricas financieras como rentabilidad anual, volatilidad diaria y anual, 
    ratio de Sharpe y ratio de Sortino para una lista de dataframes diarios.

    Args:
        dataframes_diarios (dict): Un diccionario que contiene dataframes diarios de activos financieros.
        tasa_libre_riesgo (float): Tasa libre de riesgo anual. Por defecto, 0.01.

    Returns:
        pd.DataFrame: Un DataFrame que contiene las métricas financieras calculadas para cada activo.
    """
    tasa_libre_riesgo = 0.01
    resultados = {}
    for nombre_df, df in dataframes_diarios.items():
        df2=df.copy()
        log_returns = np.log(df2['close_cubic']/df2['close_cubic'].shift())#RENDIMIENTO LOGORATIMICO
        daily_std = log_returns.std()
        annualized_std =  daily_std*np.sqrt(252)
        df2['Rentabilidad'] = df2['close_cubic'].pct_change()
        rentabilidad_anual = (1 + df2['Rentabilidad']).prod() - 1
        # volatilidad_diaria = df2['close'].std()
        # volatilidad_anual = volatilidad_diaria * (252 ** 0.5)
        # sharpe_ratio = (rentabilidad_anual - tasa_libre_riesgo) / volatilidad_anual
        sharpe_ratio2 = (rentabilidad_anual - tasa_libre_riesgo) / annualized_std
        rentabilidades_negativas = df2[df2['Rentabilidad'] < 0]['Rentabilidad']
        volatilidad_negativa_anual = (rentabilidades_negativas.std())*np.sqrt(252)
        sortino_ratio = (rentabilidad_anual - tasa_libre_riesgo) / volatilidad_negativa_anual
        resultados[nombre_df] = {'Rentabilidad Anual': rentabilidad_anual,
                                #  'Volatilidad Diaria': volatilidad_diaria,
                                #  'Volatilidad Anual': volatilidad_anual,
                                #  'Ratio de Sharpe': sharpe_ratio,
                                'Volatilidad Diaria':daily_std, 'Volatilidad Anual':annualized_std, 'Ratio de Sharpe':sharpe_ratio2,'Ratio Sortino': sortino_ratio}
    resultados_df = pd.DataFrame.from_dict(resultados, orient='index')
    return resultados_df

def calcular_metricas_financieras_2(dataframes_semanal):
    """
    Calcula las métricas financieras como rentabilidad anual, volatilidad diaria y anual, 
    ratio de Sharpe y ratio de Sortino para una lista de dataframes diarios.

    Args:
        dataframes_diarios (dict): Un diccionario que contiene dataframes diarios de activos financieros.
        tasa_libre_riesgo (float): Tasa libre de riesgo anual. Por defecto, 0.01.

    Returns:
        pd.DataFrame: Un DataFrame que contiene las métricas financieras calculadas para cada activo.
    """
    tasa_libre_riesgo = 0.01
    resultados = {}
    for nombre_df, df in dataframes_semanal.items():
        df2=df.copy()
        log_returns = np.log(df2['close_cubic']/df2['close_cubic'].shift())
        daily_std = log_returns.std()
        annualized_std =  daily_std*np.sqrt(52)
        df2['Rentabilidad'] = df2['close_cubic'].pct_change()
        rentabilidad_anual = (1 + df2['Rentabilidad']).prod() - 1
        # volatilidad_diaria = df2['close'].std()
        # volatilidad_anual = volatilidad_diaria * (252 ** 0.5)
        # sharpe_ratio = (rentabilidad_anual - tasa_libre_riesgo) / volatilidad_anual
        sharpe_ratio2 = (rentabilidad_anual - tasa_libre_riesgo) / annualized_std
        rentabilidades_negativas = df2[df2['Rentabilidad'] < 0]['Rentabilidad']
        volatilidad_negativa_anual = (rentabilidades_negativas.std())*np.sqrt(52)
        sortino_ratio = (rentabilidad_anual - tasa_libre_riesgo) / volatilidad_negativa_anual
        resultados[nombre_df] = {'Rentabilidad Anual': rentabilidad_anual,
                                #  'Volatilidad Diaria': volatilidad_diaria,
                                #  'Volatilidad Anual': volatilidad_anual,
                                #  'Ratio de Sharpe': sharpe_ratio,
                                'Volatilidad Diaria':daily_std, 'Volatilidad Anual':annualized_std, 'Ratio de Sharpe':sharpe_ratio2,'Ratio Sortino': sortino_ratio}
    resultados_df_semanal = pd.DataFrame.from_dict(resultados, orient='index')
    return resultados_df_semanal

def guardar_dataframes_csv(dataframes_diarios, directorio):
    """
    Guarda los dataframes diarios como archivos CSV en un directorio específico.

    Args:
        dataframes_diarios (dict): Un diccionario que contiene dataframes diarios de activos financieros.
        directorio (str): La ruta del directorio donde se guardarán los archivos CSV. 
    """
    # Verifica si el directorio existe, si no, créalo
    if not os.path.exists(directorio):
        os.makedirs(directorio)

    for nombre_df, df in dataframes_diarios.items():
        nombre_csv = nombre_df + '.csv'
        ruta_csv = os.path.join(directorio, nombre_csv)
        df.to_csv(ruta_csv, index=True)
