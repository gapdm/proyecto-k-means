import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from wordcloud import WordCloud

try:
    df = pd.read_csv('cuestionario.csv')
    print("\nDatos cargados correctamente:")
    print(df.head())
except FileNotFoundError:
    print("El archivo 'cuestionario.csv' no se encontró. Por favor, genere los datos primero.")
    exit()

df = df.drop_duplicates()
df = df.dropna()

df_numericos = df[['Edad', 'Prefiere_Comedia', 'Prefiere_Drama', 'Prefiere_Acción']]

print("\nEstadísticas descriptivas:")
print(df_numericos.describe())

sns.histplot(df['Edad'], bins=10, kde=True)
plt.title('Distribución de edades')
plt.xlabel('Edad')
plt.ylabel('Frecuencia')
plt.show()

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_numericos)

inertia = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 10), inertia, marker='o')
plt.title('Método del codo')
plt.xlabel('Número de clústeres')
plt.ylabel('Inercia')
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)

sns.scatterplot(data=df, x='Edad', y='Prefiere_Comedia', hue='Cluster', palette='viridis')
plt.title('Clústeres por Edad y Preferencia de Comedia')
plt.show()

X = df_numericos
y = df['Prefiere_Comedia']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))
print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))