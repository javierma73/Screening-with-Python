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
Category ![Category](https://github.com/javierma73/Screening-with-Python/blob/main/Category.png)

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
---
## Ahora vamos a visualizar la distribución de categorías:
```
from matplotlib.gridspec import GridSpec
targetCounts = resumeDataSet['Category'].value_counts()
targetLabels  = resumeDataSet['Category'].unique()
# Make square figures and axes
plt.figure(1, figsize=(25,25))
the_grid = GridSpec(2, 2)


cmap = plt.get_cmap('coolwarm')
colors = [cmap(i) for i in np.linspace(0, 1, 3)]
plt.subplot(the_grid[0, 1], aspect=1, title='CATEGORY DISTRIBUTION')

source_pie = plt.pie(targetCounts, labels=targetLabels, autopct='%1.1f%%', shadow=True, colors=colors)
plt.show()
```
resume-distribution ![This is an image](https://github.com/javierma73/Screening-with-Python/blob/main/resume-distribution.png)
## Ahora crearé una función de ayuda para eliminar las URL, los hashtags, las menciones, las letras especiales y los signos de puntuación:
```
import re
def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText
    
resumeDataSet['cleaned_resume'] = resumeDataSet.Resume.apply(lambda x: cleanResume(x))
```
### Ahora que hemos borrado el conjunto de datos, la siguiente tarea es echar un vistazo a Wordcloud. Una nube de palabras representa la mayor cantidad de palabras más grandes y viceversa:
```
import nltk
from nltk.corpus import stopwords
import string
from wordcloud import WordCloud

oneSetOfStopWords = set(stopwords.words('english')+['``',"''"])
totalWords =[]
Sentences = resumeDataSet['Resume'].values
cleanedSentences = ""
for i in range(0,160):
    cleanedText = cleanResume(Sentences[i])
    cleanedSentences += cleanedText
    requiredWords = nltk.word_tokenize(cleanedText)
    for word in requiredWords:
        if word not in oneSetOfStopWords and word not in string.punctuation:
            totalWords.append(word)
    
wordfreqdist = nltk.FreqDist(totalWords)
mostcommon = wordfreqdist.most_common(50)
print(mostcommon)

wc = WordCloud().generate(cleanedSentences)
plt.figure(figsize=(15,15))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()
```

```
[('Detalles', 484), ('Experiencia', 446), ('meses', 376), ('empresa', 330), ('descripción', 310), ('1', 290), ( 'año', 232), ('Enero', 216), ('Menos', 204), ('Datos', 200), ('datos', 192), ('Habilidad', 166), ('Maharashtra ', 166), ('6', 164), ('Python', 156), ('Science', 154), ('I', 146), ('Education', 142), ('College', 140), ('The', 126), ('project', 126), ('like', 126), ('Project', 124), ('Learning', 116), ('India', 114) , ('Máquina', 112), ('Universidad', 112), ('Web', 106), ('usando', 104), ('monthsCompany', 102), ('B', 98), ( 'C', 98), ('SQL', 96), ('tiempo', 92), ('aprendizaje', 90),('Mumbai', 90), ('Pune', 90), ('Artes', 90), ('A', 84), ('aplicación', 84), ('Ingeniería', 78), (' 24', 76), ('varios', 76), ('Software', 76), ('Responsabilidades', 76), ('Nagpur', 76), ('desarrollo', 74), ('Gestión' , 74), ('proyectos', 74), ('Tecnologías', 72)]
```
![Screening](https://github.com/javierma73/Screening-with-Python/blob/main/resume-screening-word.png)
### Ahora convertiré estas palabras en valores categóricos:
```
from sklearn.preprocessing import LabelEncoder

var_mod = ['Category']
le = LabelEncoder()
for i in var_mod:
    resumeDataSet[i] = le.fit_transform(resumeDataSet[i])
```
## Capacitación del modelo de aprendizaje automático para la selección de currículums
###### Ahora, el siguiente paso en el proceso es entrenar un modelo para la tarea de evaluación de currículums. Aquí usaré el clasificador uno contra el resto; KNeighborsClassifier. Para esta tarea, primero dividiré los datos en conjuntos de entrenamiento y prueba:
```
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

requiredText = resumeDataSet['cleaned_resume'].values
requiredTarget = resumeDataSet['Category'].values

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    stop_words='english',
    max_features=1500)
word_vectorizer.fit(requiredText)
WordFeatures = word_vectorizer.transform(requiredText)

print ("Feature completed .....")

X_train,X_test,y_train,y_test = train_test_split(WordFeatures,requiredTarget,random_state=0, test_size=0.2)
print(X_train.shape)
print(X_test.shape)
```
## Ahora entrenemos el modelo e imprimamos el informe de clasificación:
```
clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
print('Accuracy of KNeighbors Classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of KNeighbors Classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))

print("\n Classification report for classifier %s:\n%s\n" % (clf, metrics.classification_report(y_test, prediction)))
```
Precisión de KNeighbors Classifier en el conjunto de entrenamiento: 0,99 
Precisión de KNeighbors Classifier en el conjunto de prueba: 0,99 

 Informe de clasificación para el clasificador OneVsRestClassifier(estimator=KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', 
           metric_params=Ninguno, n_jobs= Ninguno, n_vecinos=5, p=2, 
           pesos='uniforme'), 
          n_trabajos=Ninguno): 
              recuperación de precisión compatibilidad con puntaje f1 

           0 1,00 1,00 1,00 3 
           1 1,00 1,00 1,00 3 
           2 1,00 0,80 0,89 5 
           3 1,00 1,00 1,00 9 
           4 1,00 1,00 1,00 6
           5 0.83 1.00 0.91 5 
           6 1.00 1.00 1.00 9 
           7 1.00 1.00 1.00 7 
           8 1.00 0.91 0.95 11 
           9 1.00 1.00 1.00 9 
          10 1.00 1.00 1.00 8 11 
          0.90 1.00 0.95 9 
          12 1.00 1.00 5 
          13 1.00 1.00 1.00 9 
          14 1.00 1.00 7 7 
          15 1,00 1,00 1,00 19 
          16 1,00 1,00 1,00 3
          17 1.00 1.00 1.00 4 
          18 1.00 1.00 1.00 5 
          19 1.00 1.00 1.00 6 20 1.00 1.00 1.00 
          11 21 1.00 1.00 1.00 
          4 
          22 1.00 1.00 1.00 1.00 1.00 1.00 15 
          24 
          1.00 1.00 1.00 8 

   Micro AVG 0.99 0.99 0.99 193 
   Macro AVG 0.99 0.99 0,99 193 
promedio ponderado 0,99 0,99 0,99 193

---
### Así es como podemos entrenar un modelo de Machine Learning para la tarea de Resume Screening. Espero que te haya gustado este artículo sobre Resume Screening con el lenguaje de programación Python. Siéntase libre de hacer sus valiosas preguntas en la sección de comentarios a continuación.
