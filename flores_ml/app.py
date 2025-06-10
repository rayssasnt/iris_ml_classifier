# carregando os dados

from sklearn.datasets import load_iris #dados
import pandas as pd

dados = load_iris()
x = pd.DataFrame(dados.data, columns=dados.feature_names) # entradas

y = dados.target #array com o nº das especies

target_names = dados.target_names # nome das especies


# 2º parte 
#treino e teste de modelos
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # função para dividir os dados em treino e teste
from sklearn.ensemble import RandomForestClassifier #modelo de ML baseado em arvores


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
##Dados dividos em 20% para teste e 80 % para treino


modelo = RandomForestClassifier()
modelo.fit(X_train, y_train)

# interface

import streamlit as st
st.title("🌸 Classificador de Espécie de Flor Iris")
st.write("Informe as características da flor para prever a espécie")

sep_len = st.slider("Comprimento da sépala (cm)", 4.0, 8.0, 5.1)
sep_wid = st.slider("Largura da sépala (cm)", 2.0, 4.5, 3.5)
pet_len = st.slider("Comprimento da pétala (cm)", 1.0, 7.0, 1.4)
pet_wid = st.slider("Largura da pétala (cm)", 0.1, 2.5, 0.2)
# sliders para definir as medidas

if st.button("🔍 Prever Espécie"):
    entrada = [[sep_len , sep_wid , pet_len , pet_wid]]
    predicao = modelo.predict(entrada)[0]
    especie = target_names[predicao]

    st.success(f"A flor provavelmente é da espécie **{especie.capitalize()}**")

    st.markdown("### 📷 Imagem da Flor")
    import os

    caminho_imagem = os.path.join(os.path.dirname(__file__), "imgs", f"{especie}.jpeg")
    st.image(caminho_imagem, caption=f"Flor da espécie: {especie.capitalize()}", use_container_width=True)


