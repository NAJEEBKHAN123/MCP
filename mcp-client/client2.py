import asyncio
import concurrent.futures
import json
import os
import threading
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import (
    ChatGoogleGenerativeAIError,
    GoogleAPIError,
)
from langchain_mcp_adapters.client import MultiServerMCPClient

from client1 import calculate, get_servers

load_dotenv()

SYSTEM_PROMPT = (
    "Answer the user clearly and concisely. Use the available tools whenever "
    "needed to perform calculations or manage/query expenses. Do not claim an "
    "expense was saved unless the expense tool confirms success."
)
MODEL_NAME = "gemini-3.8-flash"
REQUEST_TIMEOUT_SECONDS = 35
MAX_TOOL_ROUNDS = 5


class AsyncLoopRunner:
    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="mcp-async-loop",
            daemon=True,
        )
        self._thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, awaitable: Any) -> Any:
        future = asyncio.run_coroutine_threadsafe(awaitable, self._loop)
        try:
            return future.result()
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise TimeoutError from None


@st.cache_resource
def get_async_runner() -> AsyncLoopRunner:
    return AsyncLoopRunner()


def display_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            item["text"]
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ]
        return "\n".join(parts) if parts else json.dumps(content, default=str)
    return str(content)


st.set_page_config(page_title="Gemini MCP Chat", page_icon="🧰")
load_dotenv()

st.title("🧰 Gemini MCP Chat")
st.caption("Chat with Gemini, use local calculations, and manage expenses.")

api_key = os.getenv("GEMINI_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Set GEMINI_KEY or GOOGLE_API_KEY in your .env file, then restart the app.")
    st.stop()

if "mcp_chat_initialized_v2" not in st.session_state:
    try:
        st.session_state.async_runner = get_async_runner()
        tools = [calculate]
        servers = get_servers()
        if servers:
            mcp_client = MultiServerMCPClient(servers)
            tools.extend(st.session_state.async_runner.run(mcp_client.get_tools()))

        st.session_state.mcp_tools = {tool.name: tool for tool in tools}
        st.session_state.chat_model = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            api_key=api_key,
            request_timeout=30,
            retries=0,
        )
        st.session_state.chat_history = [
            SystemMessage(content=SYSTEM_PROMPT)
        ]
        st.session_state.mcp_chat_initialized_v2 = True
    except Exception as exc:
        st.error(f"Could not initialize the MCP tools: {type(exc).__name__}: {exc}")
        st.stop()

with st.sidebar:
    st.subheader("Available tools")
    for tool_name in st.session_state.mcp_tools:
        st.write(f"- `{tool_name}`")
    if st.button("Clear conversation"):
        st.session_state.chat_history = [SystemMessage(content=SYSTEM_PROMPT)]
        st.rerun()

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(display_content(message.content))
    elif isinstance(message, AIMessage) and not message.tool_calls:
        with st.chat_message("assistant"):
            st.markdown(display_content(message.content))

user_text = st.chat_input("Ask about an expense or a calculation...")
if user_text:
    history = st.session_state.chat_history
    history.append(HumanMessage(content=user_text))
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        response_area = st.empty()
        status_area = st.status("Thinking...", expanded=False)
        attempted_tools: list[str] = []
        try:
            llm_with_tools = st.session_state.chat_model.bind_tools(
                list(st.session_state.mcp_tools.values())
            )
            for _ in range(MAX_TOOL_ROUNDS):
                response = st.session_state.async_runner.run(
                    asyncio.wait_for(
                        llm_with_tools.ainvoke(history),
                        timeout=REQUEST_TIMEOUT_SECONDS + 1,
                    )
                )

                if not response.tool_calls:
                    answer = display_content(response.content)
                    history.append(response)
                    response_area.markdown(answer or "Gemini returned an empty response.")
                    status_area.update(label="Done", state="complete", expanded=False)
                    break

                history.append(response)
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool = st.session_state.mcp_tools.get(tool_name)
                    if tool is None:
                        raise RuntimeError(f"Gemini requested an unavailable tool: {tool_name}")

                    attempted_tools.append(tool_name)
                    status_area.update(label=f"Running {tool_name}...")
                    try:
                        result = st.session_state.async_runner.run(
                            tool.ainvoke(tool_call.get("args") or {})
                        )
                        result_content = json.dumps(result, default=str)
                    except Exception as exc:
                        result_content = json.dumps(
                            {"error": f"{type(exc).__name__}: {exc}"}
                        )
                        st.error(f"Tool `{tool_name}` failed: {exc}")

                    history.append(
                        ToolMessage(
                            tool_call_id=tool_call["id"],
                            content=result_content,
                        )
                    )
            else:
                raise RuntimeError(
                    f"Stopped after {MAX_TOOL_ROUNDS} tool rounds to prevent an infinite loop."
                )
        except TimeoutError:
            status_area.update(label="Gemini request timed out", state="error")
            if attempted_tools:
                message = (
                    "Gemini timed out after attempting tool(s): "
                    f"{', '.join(attempted_tools)}. A tool may already have "
                    "completed; verify expenses before retrying to avoid duplicates."
                )
            else:
                message = (
                    "Gemini did not respond in time, so no tool was called. "
                    "Check internet access, model availability, and quota in "
                    "Google AI Studio."
                )
            history.append(AIMessage(content=message))
            st.error(message)
        except (ChatGoogleGenerativeAIError, GoogleAPIError) as exc:
            status_area.update(label="Gemini API error", state="error")
            if attempted_tools:
                message = (
                    f"Gemini API error ({type(exc).__name__}) after tool "
                    f"attempt(s): {', '.join(attempted_tools)}. Check the "
                    "expense list before retrying to avoid duplicates. "
                    f"Details: {exc}"
                )
            else:
                message = (
                    f"Gemini API error ({type(exc).__name__}): {exc}. "
                    "No tool was called."
                )
            history.append(AIMessage(content=message))
            st.error(message)
        except Exception as exc:
            status_area.update(label="Request failed", state="error")
            message = f"Request failed ({type(exc).__name__}): {exc}"
            history.append(AIMessage(content=message))
            st.error(message)
