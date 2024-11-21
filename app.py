import os
import base64
from io import BytesIO

import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from st_multimodal_chatinput import multimodal_chatinput
from streamlit_extras.tags import tagger_component
from dataclasses import dataclass

from src.conversation_manager import ConversationManager
from src.agent import QAAgent

st.set_page_config(
    page_title="Open-Omnisearch Demo",
    page_icon="🤖",
    layout="wide"
)

st.title("open-omnisearch Demo 🤖")
st.markdown("📦 More details can be found at the GitHub repo [here](https://github.com/KKenny0/Open-OmniSearch)")


@dataclass
class Message:
    actor: str
    payload: str
    image: str


os.makedirs("search_multimodal_chat", exist_ok=True)
conversation_manager = ConversationManager(qa_agent=QAAgent(),
                                           dataset_name="ui-chat",
                                           save_path="search_multimodal_chat")

USER = "user"
ASSISTANT = "ai"
MESSAGES = "messages"

if MESSAGES not in st.session_state:
    st.session_state[MESSAGES] = []


col1, col2 = st.columns(2)
with col1.container(border=True,):
    col1_message = col1.container(height=800)
    col1_message.chat_message(ASSISTANT).write("不如问我点什么吧~😄")
    msg: Message
    for msg, image in st.session_state[MESSAGES]:
        with col1_message.chat_message(msg.actor):
            col1_message.write(msg.payload)
            if image:
                col1_message.image(image)


with col2.container(border=True, height=800):
    tagger_component("思考过程", ["🤔"])


row2_col1, row2_col2 = st.columns(2)
with row2_col1.container():
    chat_input = multimodal_chatinput()

    if chat_input:
        uploaded_images = chat_input["images"]  # list of base 64 encoding of uploaded images
        prompt = chat_input["text"]  # submitted text

        if len(uploaded_images) > 0:
            image = BytesIO(base64.b64decode(uploaded_images[0].split("base64,")[-1]))
        else:
            image = ""
        st.session_state[MESSAGES].append(Message(actor=USER, payload=prompt, image=image))

        col1_message.chat_message(USER).write(prompt)
        if image:
            col1_message.chat_message(USER).image(image)

        with st.spinner("Please wait.."):
            answer, _ = conversation_manager.manage_conversation(
                input_question=prompt,
                image_url=image,
                idx="test"
            )
            st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload=answer, image=""))
            col1_message.chat_message(ASSISTANT).write(answer)
