import json
import streamlit as st
import requests

with st.spinner():
    endpoint = "http://172.18.1.55//lista_contexto"
    r = requests.get(url=endpoint).json()
    try:
        lista_con = r
        lista_id = [f"{idcon['IdContexto']} - {idcon['Titulo']}" for idcon in r]
    except TypeError:
        lista_con = r
        lista_id = []


option = st.selectbox(
    'Escolha o contexto ou a opção de escrever um novo',
    lista_id + ['Escrever']
)

with st.container():
    if option != 'Escrever':
        with st.spinner():
            endpoint = "http://172.18.1.20/contexto"
            data = json.dumps({"IdContexto": option.split(" -")[0]})
            headers = {"Authorization": "Bearer 3ef78c01c63c9834b505fe22877e2b5a23bc0fcc624e53c7377bd652164cc216"}
            r = requests.post(url=endpoint, data=data, headers=headers)
            r = r.json()
        st.write(r["Contexto"])
    else:
        titulo = st.text_input("Titulo:")
        contexto = st.text_area('Contexto:', height=80)


pergunta = st.text_input("Pergunta")

col1, col2, col3, col4, col5 = st.columns(5, gap="small")

button_pergunta = col1.button("Enviar")

button_salva = col2.button("Salvar no BD") if option == 'Escrever' else False


with st.spinner("Recebendo resposta.."):
    if button_pergunta and pergunta:
        if option != 'Escrever':
            endpoint = "http://172.18.1.20/pergunta_db"
            data = json.dumps({'Pergunta': pergunta, 'IdContexto':  option.split(" -")[0]})
        else:
            endpoint = "http://172.18.1.20/pergunta_contexto"
            data = json.dumps({'Pergunta': pergunta, 'Contexto': contexto})

        headers = {"Authorization": "Bearer 3ef78c01c63c9834b505fe22877e2b5a23bc0fcc624e53c7377bd652164cc216"}
        r = requests.post(url=endpoint, data=data, headers=headers)
        r = r.json()
        st.write("Resposta")
        st.write(r['Resposta'])
        st.write("Detalhes da requisição")
        st.write(r)

    if button_salva and contexto and titulo:
        endpoint = "http://172.18.1.20/add_contexto"
        data = json.dumps({'Contexto': contexto, 'Titulo': titulo})
        headers = {"Authorization": "Bearer 3ef78c01c63c9834b505fe22877e2b5a23bc0fcc624e53c7377bd652164cc216"}
        r = requests.post(url=endpoint, data=data, headers=headers)
        r = r.json()
        st.write("Contexto salvo")
        st.write("Detalhes da requisição")
        st.write(r)

