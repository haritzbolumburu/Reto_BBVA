# RETO09_NARANJA
Este repositorio de GitHub contiene todo el trabajo realizado por el equipo Naranja desde 18/09/2023 hasta 10/11/2023. Este proyecto da respuesta a las dos problemáticas presentadas por BBVA. Para la correcta ejecución del trabajo habrá que descargar los datos y guardarlos en la carpeta 'Datos/Originales. '

# Tecnologías y Lenguajes de Programación utilizadas
Para resolver los objetivos se ha hecho uso de dos tecnologías. Por una parte, a lo largo de todo el proyecto predomina el lenguaje de programación de Python. Por otro lado, se ha utilizado Mlflow una plataforma que sirve para administrar el ciclo de vida de los modelos creados. 

# Entorno virtual:
Antes de empezar a ejecutar los scripts del proyecto habrá que instalar el entorno virtual que contiene todas las librerías utilizadas. Para ello habrá que instalar el archivo .yml que contiene el proyecto a través de los siguientes comandos:
```
> conda env create -f environment_reto.yml
> conda activate reto9_bbva
```

# Estructura carpetas:
El proyecto está compuesto por dos carpetas Packages y Datos. 

- Packages: en esta carpeta se encuentran las funciones necesarias para cargar los ficheros .py. 
- Datos: en esta carpeta se encontrarán los datos originales en una sub carpeta y los transformados en otra sub carpeta.

# Estructura y Scripts:
El repositorio del trabajo contiene dos tipos de ficheros: .py y .ipynb. Los ficheros .py se ejecutarán desde la terminal de Anaconda, en cambio, los .ipynb en Visual Studio Code. 
El proyecto se ejecutará y tendrá la siguiente estructura.

**1. Preprocesamiento**. En este apartado se procederá a cargar, analizar y adecuar los datos entregados por BBVA para la resolución de los objetivos.
```
python 01-Ingesta_Limpieza.py
```
**2. Análisis Financiero**. Es un notebook donde se realiza un análisis financiero sobre todas las acciones y se procede a la elección de los activos que componen nuestra cartera. 
```
Analisis_financiero.ipynb
```
**3. Análisis Cartera**. Notebook donde se analizan los activos seleccionados de la cartera para valorar las componentes de las series temporales.
```
analisis_activos_seleccionados.ipynb
```
**4. Montecarlo**. En este notebook/ script se realiza las simulaciones de los activos mediante el método Montecarlo.
```
NOMBRE SCRIPT
```
**5. Modelado - Red Nueronal**. En este script se realizará la creación de una red neuronal recurrente y el entrenamiento de esta. 
```
ptyhon 04-Modelos.py
```

# Autores:
Mikel Gonzalez, Peio Garcia, Uxue Uribe, Haritz Bolomburu y Ainhize Barredo. 
