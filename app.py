import io

import os

import streamlit as st
from streamlit_extras.tags import tagger_component
from dataclasses import dataclass

from src.conversation_manager import ConversationManager
from src.agent import QAAgent

st.set_page_config(
    # page_title="Open-Omnisearch Demo",
    page_title="Omnisearch Demo",
    page_icon="🤖",
    layout="wide"
)

st.title("omnisearch Demo 🤖")
# st.title("open-omnisearch Demo 🤖")
# st.markdown("📦 More details can be found at the GitHub repo [here](https://github.com/KKenny0/Open-OmniSearch)")


@dataclass
class Message:
    actor: str
    payload: str
    image: str


os.makedirs("search_multimodal_chat", exist_ok=True)

USER = "user"
ASSISTANT = "ai"
MESSAGES = "messages"

if MESSAGES not in st.session_state:
    st.session_state[MESSAGES] = []


with st.sidebar:
    st.subheader("LLM 参数")
    st.session_state['OLLAMA_HOST'] = st.text_input("输入 Ollama 服务 host",
                                                    value="http://localhost:11434")
    st.session_state['OLLAMA_MODEL'] = st.text_input("输入 Ollama 模型名称",
                                                     value="llama3.2-vision:11b-instruct-q4_K_M")
    st.session_state['LLM_TEMPERATURE'] = st.slider("**Temperature**",
                                                    value=0.1,
                                                    min_value=0.0,
                                                    max_value=1.0,
                                                    step=0.1)
    st.session_state['LLM_TOP_P'] = st.slider("**Top P**",
                                              value=1.0,
                                              min_value=0.0,
                                              max_value=1.0,
                                              step=0.1)

    st.session_state['client'] = ConversationManager(qa_agent=QAAgent(st.session_state['OLLAMA_HOST'],
                                                                      st.session_state['OLLAMA_MODEL'],
                                                                      temperature=st.session_state['LLM_TEMPERATURE'],
                                                                      top_p=st.session_state['LLM_TOP_P']
                                                                      ),
                                                     dataset_name="ui-chat",
                                                     save_path="search_multimodal_chat")

col1, col2 = st.columns(2)
with col1.container(border=True,):
    col1_message = col1.container(height=800)
    col1_message.chat_message(ASSISTANT).write("不如问我点什么吧~😄")
    msg: Message
    for msg in st.session_state[MESSAGES]:
        with col1_message.chat_message(msg.actor):
            col1_message.write(msg.payload)
            if msg.image:
                col1_message.image(msg.image)


with col2.container(border=True,):
    tagger_component("思考过程", ["🤔"])
    col2_message = col2.container(height=720)


row2_col1, row2_col2 = st.columns(2)
with row2_col1.container():
    uploaded_image = st.file_uploader("Choose an image for Multimodal QA", type=['png', 'jpg', 'jpeg'])
    prompt = st.chat_input("Query something")

if prompt:
    if uploaded_image is not None:
        image = io.BytesIO(uploaded_image.getvalue())
    else:
        image = ""
    st.session_state[MESSAGES].append(Message(actor=USER, payload=prompt, image=image))

    col1_message.chat_message(USER).write(prompt)
    if image:
        col1_message.chat_message(USER).image(image)

    with st.spinner("Please wait.."):
        answer, _, thought_info = st.session_state['client'].manage_conversation(
            input_question=prompt,
            image_url=image,
            idx="test"
        )
        st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload=answer, image=""))
        col1_message.chat_message(ASSISTANT).write(answer)

    for idx, (thought, search, sub_q) in enumerate(zip(thought_info['thoughts'],
                                                       thought_info['search'],
                                                       thought_info['sub_questions'])):
        col2_message.write(f"🤔🤔行动链 {idx + 1}")
        col2_message.write(f":red[思考: {thought}]")
        if "http" in search[:5]:
            col2_message.write(f":green[搜索动作: 图片搜索]")
            col2_message.image(search)
        else:
            col2_message.write(f":green[搜索动作: 搜索文字]")
            col2_message.write(search)
        col2_message.write(f":blue[子问题: {sub_q}]")
