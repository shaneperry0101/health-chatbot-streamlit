import streamlit as st

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

import ast

from core.agents import Agent
from core.tools import TavilySearch, VideoSearch
from utils import video_dialog, search_dialog


# Page configuration
st.set_page_config(
    page_title="Healthcare Assistant",
    page_icon="./public/favicon.ico",
    layout="centered",
    menu_items={
        "Report a bug": "https://github.com/shaneperry0102/health-chatbot/issues/new?title=New%20Bug&body=Here%20is%20a%20bug%20detail.",
        "About": "## This is an *extremely* cool healthcare chatbot!"
    },
    initial_sidebar_state="auto"
)


provider_dict = {
    'OpenAI': {
        'models': ['gpt-4o-mini', 'gpt-4o'],
        'api_key': 'OPENAI_API_KEY'
    },
    'Groq': {
        'models': ['llama3-70b-8192'],
        'api_key': 'GROQ_API_KEY'
    },
    'AI21': {
        'models': ['jamba-1.5-large', 'jamba-1.5-mini'],
        'api_key': 'AI21_API_KEY'
    },
    'HuggingFace': {
        'models': ['HuggingFaceH4/zephyr-7b-beta'],
        'api_key': 'HUGGINGFACE_API_KEY'
    }
}


# Sidebar
with st.sidebar:
    bar = st.container()

    st.title(":stethoscope: Healthcare Assistant")
    st.write('This chatbot is created using the LLM model.')

    st.subheader('Models and parameters')
    selected_provider = st.sidebar.selectbox(
        label='Choose a provider',
        options=['OpenAI', 'Groq', 'AI21', 'HuggingFace'],
        key='selected_provider'
    )
    selected_model = st.sidebar.selectbox(
        label='Choose a model',
        options=provider_dict[selected_provider]["models"],
        key='selected_model'
    )
    temperature = st.sidebar.slider(
        label='temperature',
        min_value=0.01,
        max_value=2.0,
        value=1.0,
        step=0.01
    )

    if provider_dict[selected_provider]["api_key"] in st.secrets:
        st.success('API key already provided!', icon='✅')
        api_key_name = provider_dict[selected_provider]["api_key"]
        api_key = st.secrets[api_key_name]
    else:
        api_key = st.text_input(
            'Enter API token:', type='password')
        if not api_key:
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')

    st.write(
        "[View the source code](https://github.com/shaneperry0102/health-chatbot/blob/main/app.py)")
    st.write("[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/shaneperry0102/health-chatbot?quickstart=1)")


def init_chat_history():
    """Initialize chat history"""
    st.session_state["messages"] = []
    st.session_state["chat_history"] = []


if "chat_history" not in st.session_state:
    init_chat_history()

bar.button('Clear Chat History', on_click=init_chat_history)


system_prompt = """You are a smart healthcare assistant. \
    You should first check if the questions are related to healthcare. \
    If not, do not answer, just refuse it, e.g. "Sorry, I can't answer that question." \
    Use the search engine to look up information if needed. \
    You are allowed to make multiple calls (either together or in sequence). \
    Only look up information when you are sure of what you want. \
    If you need to look up some information before asking a follow up question, \
    you are allowed to do that!
"""


def create_agent(tools: list, system=system_prompt):
    model = ChatOpenAI(
        api_key=api_key,
        model=selected_model,
        temperature=temperature,
        streaming=True
    )
    # model = ChatGroq(
    #     api_key=api_key,
    #     model=selected_model,
    #     temperature=temperature,
    #     max_tokens=None,
    #     timeout=None,
    #     max_retries=2,
    #     streaming=True
    # )
    agent = Agent(model=model, tools=tools, system=system)
    return agent


messages = st.container()

# Display chat messages from history on app rerun
for message in st.session_state["chat_history"]:
    if message["role"] == "user":
        messages.chat_message("user").write(message["content"]["text"])
    elif message["role"] == "assistant":
        assistant_chat = messages.chat_message("assistant")
        for part in message["content"]:
            if part["type"] == "text":
                assistant_chat.markdown(part["text"])
            elif part["type"] == "youtube":
                if assistant_chat.button("Videos", key=part["list"]):
                    video_dialog(part["list"])
            elif part["type"] == "search":
                if assistant_chat.button("Sources", key=part["list"]):
                    search_dialog(part["list"])


# React to user input
if user_prompt := st.chat_input(placeholder="Message Here", disabled=not api_key):
    # Display user message in chat message container
    messages.chat_message("user").write(user_prompt)
    # Add user message to chat history
    st.session_state["messages"].append(HumanMessage(content=user_prompt))
    st.session_state["chat_history"].append(
        {"role": "user", "content": {"type": "text", "text": user_prompt}})

    thread = {"configurable": {"thread_id": "4"}}
    aimessages = ""
    health_graph = create_agent(tools=[TavilySearch]).graph
    video_graph = create_agent(tools=[VideoSearch]).graph

    tool_text_list = []
    searchResults = None

    with messages.chat_message("assistant"):
        with st.spinner("Thinking..."):
            for event in health_graph.stream(
                input={"messages": st.session_state["messages"]},
                config=thread,
                stream_mode="values"
            ):
                message = event["messages"][-1]
                if isinstance(message, AIMessage) and message.content:
                    aimessages += str(message.content)
                    tool_text_list.append(
                        {"type": "text", "text": message.content})
                    st.markdown(message.content)
                elif isinstance(message, ToolMessage) and message.name == TavilySearch.name:
                    searchResults = ast.literal_eval(message.content)

            if searchResults and len(searchResults):
                tool_text_list.append(
                    {"type": "search", "list": searchResults})
                if st.button("Sources", key=searchResults):
                    search_dialog(searchResults)

            for event in video_graph.stream(
                input={"messages": st.session_state["messages"]},
                config=thread,
                stream_mode="values"
            ):
                message = event["messages"][-1]
                if isinstance(message, ToolMessage) and message.name == VideoSearch.name:
                    results = ast.literal_eval(message.content)
                    if len(results):
                        tool_text_list.append(
                            {"type": "youtube", "list": results})
                        if st.button("Videos", key=results):
                            video_dialog(results)

    st.session_state["messages"].append(AIMessage(content=aimessages))
    st.session_state["chat_history"].append(
        {"role": "assistant", "content": tool_text_list})
