# Screening-with-Python
Las empresas a menudo reciben miles de currículos para cada puesto de trabajo y emplean funcionarios de selección dedicados para evaluar a los candidatos calificados. En este artículo, le presentaré un proyecto de aprendizaje automático sobre la evaluación de currículums con el lenguaje de programación Python
## ¿Qué es la selección de currículums?
Contratar el talento adecuado es un desafío para todas las empresas. Este desafío se ve magnificado por el alto volumen de solicitantes si el negocio requiere mucha mano de obra, está en crecimiento y enfrenta altas tasas de deserción.
Un ejemplo de un negocio de este tipo es que los departamentos de TI no están a la altura de los mercados en crecimiento. En una organización de servicios típica, se contratan y asignan a proyectos para resolver los problemas de los clientes profesionales con una variedad de habilidades técnicas y experiencia en el dominio comercial. Esta tarea de seleccionar el mejor talento entre muchos otros se conoce como Resume Screening.
Por lo general, las grandes empresas no tienen suficiente tiempo para abrir cada CV, por lo que utilizan algoritmos de aprendizaje automático para la tarea de evaluación de currículums.
## Proyecto de aprendizaje automático sobre selección de currículums con Python
En esta sección, lo guiaré a través de un proyecto de aprendizaje automático sobre la selección de currículums con el lenguaje de programación Python. Comenzaré esta tarea importando las bibliotecas de Python necesarias y el conjunto de datos:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

resumeDataSet = pd.read_csv('UpdatedResumeDataSet.csv' ,encoding='utf-8')
resumeDataSet['cleaned_resume'] = ''
resumeDataSet.head()
```
Category ![Category](https://github.com/javierma73/Screening-with-Python/blob/main/resume-2.JPG)

Ahora echemos un vistazo rápido a las categorías de currículos presentes en el conjunto de datos:
```
print ("Displaying the distinct categories of resume -")
print (resumeDataSet['Category'].unique())
```
---
[Download Dataset](https://www.kaggle.com/dhainjeamita/updatedresumedataset/download)
```
Displaying the distinct categories of resume -
['Data Science' 'HR' 'Advocate' 'Arts' 'Web Designing'
 'Mechanical Engineer' 'Sales' 'Health and fitness' 'Civil Engineer'
 'Java Developer' 'Business Analyst' 'SAP Developer' 'Automation Testing'
 'Electrical Engineering' 'Operations Manager' 'Python Developer'
 'DevOps Engineer' 'Network Security Engineer' 'PMO' 'Database' 'Hadoop'
 'ETL Developer' 'DotNet Developer' 'Blockchain' 'Testing']
```
---
```
Displaying the distinct categories of resume and the number of records belonging to each category -
Java Developer               84
Testing                      70
DevOps Engineer              55
Python Developer             48
Web Designing                45
HR                           44
Hadoop                       42
Mechanical Engineer          40
Sales                        40
ETL Developer                40
Blockchain                   40
Operations Manager           40
Data Science                 40
Arts                         36
Database                     33
Electrical Engineering       30
Health and fitness           30
PMO                          30
DotNet Developer             28
Business Analyst             28
Automation Testing           26
Network Security Engineer    25
SAP Developer                24
Civil Engineer               24
Advocate                     20
Name: Category, dtype: int64
```
Ahora visualicemos el número de categorías en el conjunto de datos:
```
import seaborn as sns
plt.figure(figsize=(15,15))
plt.xticks(rotation=90)
sns.countplot(y="Category", data=resumeDataSet)
```
![This is an image](https://github.com/javierma73/Screening-with-Python/blob/main/resume-2.JPG)
