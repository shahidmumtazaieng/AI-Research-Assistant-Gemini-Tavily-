# app.py
import datetime
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_tavily import TavilySearch, TavilyExtract
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage

# Load environment variables
load_dotenv()

# ---------------- LLM ----------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.5,
    max_retries=2,
    timeout=60,
    model_kwargs={"streaming": True},  # ✅ correct streaming
)

# ---------------- Tools ----------------
tavily_search = TavilySearch(max_results=5, topic="general")
tavily_extract = TavilyExtract()

tools = [
    Tool(
        name="TavilySearch",
        func=tavily_search.run,
        description=(
            "Search the web for **latest, up-to-date information** on any topic. "
            "Always use this for current events or recent data."
        ),
    ),
    Tool(
        name="TavilyExtract",
        func=tavily_extract.run,
        description=(
            "Extract insights from a given URL or long text. "
            "Always use this when the user provides a link or document."
        ),
    ),
]

# ---------------- Memory ----------------
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ---------------- System Prompt ----------------
today = datetime.datetime.today().strftime("%Y-%m-%d")
system_message = SystemMessage(
    content=(
        f"You are a professional research assistant. Today's date is {today}. "
        "If the user asks for **current, real-time, or recent information**, "
        "you must use TavilySearch to gather updated details instead of relying only on your own knowledge. "
        "Use TavilyExtract when analyzing URLs or long passages of text. "
        "Always provide structured summaries, cite sources if available, and ensure responses are factually grounded."
    )
)

# ---------------- Agent ----------------
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True,
)

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="AI Research Assistant", page_icon="🔎", layout="wide")
st.title("🔎 AI Research Assistant (Gemini + Tavily)")

# Session state for chat messages
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": system_message.content}]

# Render chat history
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# Chat input
if user_input := st.chat_input("Ask me anything about research, tech, or news..."):
    # Save user input
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant response (streaming + spinner)
    with st.chat_message("assistant"):
        with st.spinner("🔎 Searching & thinking..."):
            response_placeholder = st.empty()
            response_text = ""

            try:
                for event in agent.stream({"input": user_input}):
                    if "output" in event:  # final output
                        response_text = event["output"]
                        response_placeholder.markdown(response_text)
                    elif "intermediate_steps" in event:  # optional debug info
                        pass
            except Exception as e:
                response_text = f"⚠️ Error: {str(e)}"
                response_placeholder.markdown(response_text)

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": response_text})
