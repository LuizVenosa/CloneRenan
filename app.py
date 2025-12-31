import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from brain import agent  # Importando o agente do outro arquivo

st.set_page_config(page_title="Renan Santos AI", layout="centered")

st.title("Renan Santos AI")
st.markdown("---")

# Inicialização do Histórico
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibição do histórico de mensagens
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# Entrada do Usuário
if prompt := st.chat_input("Diga algo para o Renan..."):
    # Salva e mostra a mensagem do usuário
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Geração da resposta
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
       
        # O spinner dá o feedback visual de processamento
        with st.spinner("O clone está processando..."):
            inputs = {"messages": st.session_state.messages}
            
            for chunk in agent.stream(inputs):
                for node, values in chunk.items():
                    if node == "chatbot":
                        # values["messages"][-1] é o objeto da mensagem
                        msg_objeto = values["messages"][-1]
                        
                        # ACESSE APENAS O CONTEÚDO (evita o JSON e a signature)
                        if hasattr(msg_objeto, 'content') and msg_objeto.content:
                            full_response = msg_objeto.content[0]['text']
                            placeholder.markdown(full_response)
        
        # Salva a resposta da IA no histórico
        if full_response:
            st.session_state.messages.append(AIMessage(content=full_response))