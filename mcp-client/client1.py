import asyncio
import json
import os
import sys
import shutil
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import (
    ChatGoogleGenerativeAIError,
    GoogleAPIError,
)
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()


@tool
def calculate(
    operation: Literal["add", "subtract", "multiply", "divide", "power"],
    a: float,
    b: float,
) -> float:
    """Calculate a basic arithmetic operation locally on this machine."""
    if operation == "add":
        return a + b
    if operation == "subtract":
        return a - b
    if operation == "multiply":
        return a * b
    if operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero.")
        return a / b
    return a**b


def get_servers() -> dict[str, dict[str, object]]:
    expense_server_path = os.getenv("EXPENSE_MCP_PATH")
    if not expense_server_path:
        default_path = (
            Path(__file__).resolve().parent.parent
            / "fastmcp-demo-server"
            / "main.py"
        )
        if default_path.is_file():
            expense_server_path = str(default_path)

    if expense_server_path:
        server_path = Path(expense_server_path).expanduser().resolve()
        if not server_path.is_file():
            raise FileNotFoundError(f"Expense MCP server not found: {server_path}")
        uv_executable = shutil.which("uv")
        if uv_executable is None:
            raise RuntimeError("The `uv` executable is required to launch the expense MCP server.")
        return {
            "expense": {
                "transport": "stdio",
                "command": uv_executable,
                "args": [
                    "run",
                    "--directory",
                    str(server_path.parent),
                    "fastmcp",
                    "run",
                    str(server_path),
                ],
            }
        }

    expense_url = os.getenv("EXPENSE_MCP_URL")
    if not expense_url:
        return {}

    expense_server: dict[str, object] = {
        "transport": "streamable_http",
        "url": expense_url,
    }
    expense_token = os.getenv("EXPENSE_MCP_TOKEN")
    if expense_token:
        expense_server["headers"] = {"Authorization": f"Bearer {expense_token}"}

    return {"expense": expense_server}


async def main():
    api_key = os.getenv("GEMINI_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_KEY or GOOGLE_API_KEY to use the Gemini API.")

    servers = get_servers()
    tools = [calculate]
    if servers:
        client = MultiServerMCPClient(servers)
        tools.extend(await client.get_tools())

    named_tools = {}
    for tool in tools:
        named_tools[tool.name] = tool

    print("Available tools:", named_tools.keys(), flush=True)
    if not any(name != "calculate" for name in named_tools):
        print(
            "Expense tools are not configured. Set EXPENSE_MCP_PATH to your "
            "local expense server's main.py.",
            flush=True,
        )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        api_key=api_key,
        request_timeout=30,
        retries=0,
    )
    llm_with_tools = llm.bind_tools(tools)

    try:
        while True:
            prompt = input("What would you like to ask? ").strip()
            if prompt:
                break

        messages = [HumanMessage(content=prompt)]
        for _ in range(5):
            print("Sending request to Gemini...", flush=True)
            try:
                response = await asyncio.wait_for(
                    llm_with_tools.ainvoke(messages),
                    timeout=35,
                )
            except TimeoutError:
                print(
                    "Gemini timed out after 35 seconds. No expense tool was "
                    "called and no expense was recorded. The Google API host "
                    "is reachable, but this does not verify your API key, "
                    "model access, or quota. Check those in Google AI Studio.",
                    file=sys.stderr,
                    flush=True,
                )
                return
            except (ChatGoogleGenerativeAIError, GoogleAPIError) as exc:
                print(
                    f"Gemini API error ({type(exc).__name__}): {exc}\n"
                    "No expense tool was called. Check the API key, model "
                    "access, and quota in Google AI Studio.",
                    file=sys.stderr,
                    flush=True,
                )
                return

            if not getattr(response, "tool_calls", None):
                print(f"\nResponse: {response.content}")
                return

            messages.append(response)
            for tool_call in response.tool_calls:
                selected_tool = tool_call["name"]
                if selected_tool not in named_tools:
                    raise RuntimeError(f"Gemini requested an unknown tool: {selected_tool}")

                print(f"Running tool: {selected_tool}", flush=True)
                result = await named_tools[selected_tool].ainvoke(
                    tool_call.get("args") or {}
                )
                messages.append(
                    ToolMessage(
                        tool_call_id=tool_call["id"],
                        content=json.dumps(result, default=str),
                    )
                )

        raise RuntimeError("Stopped after 5 tool-call rounds to prevent an infinite loop.")
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.", flush=True)


if __name__ == '__main__':
    asyncio.run(main())