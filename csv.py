#Faz leitura do arquivoCSV usando pandas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

dataset = pd.read_csv(r'C:\Users\pedro\Desktop\04_dados_exercicio.csv') 


#Cria uma base contendo as variáveis independentes e uma base contendo a variável dependente.
features = dataset.iloc[:, :-1].values # variáveis independentes
classe =  dataset.iloc[:, -1].values # variáveis dependente

#Substitui dados faltantes pela média da respectiva variável
 # a tabela não contém dados faltantes


#Codifica todas as variáveis categóricas independentes com One Hot Enco-ding.
ColumnTransformer = ColumnTransformer(
    transformers = [('encoder', OneHotEncoder(), [0])],
           remainder = 'passthrough')
    features =
          np.array(ColumnTransformer.fit_transform(features))

    labelEncoder = labelEncoder()
    classe = labelEncoder.fit_transform(classe)

#Codifica a variável dependente com codificação por rótulo
labelEncoder = labelEncoder()
    classe = labelEncoder.fit_transform(classe)



#Separa a base em duas partes:uma para treinamento e outra para testes. Use 85% das instâncias para o treinamento
features_treinamento, features_teste, classe_treinamento, 
    classe_teste = train_test_split(features, classe, test_size = 0.85, random_state = 1)