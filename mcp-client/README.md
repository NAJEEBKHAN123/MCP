# MCP client

This client uses Gemini to select tools, runs arithmetic locally in Python,
and can connect to a local or remote expense MCP server.

## Configuration

Create or update `.env` with:

```dotenv
GEMINI_KEY=your-gemini-api-key
# Optional if the sibling fastmcp-demo-server folder is not available:
EXPENSE_MCP_PATH=/absolute/path/to/expense-server/main.py
# Or use a remote server instead:
# EXPENSE_MCP_URL=https://your-expense-server.example/mcp
# EXPENSE_MCP_TOKEN=your-bearer-token
```

`GOOGLE_API_KEY` can be used instead of `GEMINI_KEY`. No separate math server
is required. If the `fastmcp-demo-server` folder is next to this project, its
local `main.py` is detected automatically. Otherwise, set `EXPENSE_MCP_PATH`.
For a remote expense server, set `EXPENSE_MCP_URL` and, if required,
`EXPENSE_MCP_TOKEN`.

Run the client with:

```sh
uv run python client1.py
```

For the Streamlit chat interface, run:

```sh
uv run streamlit run client2.py
```

In either client, ask a question such as "Summarize my expenses from
2026-01-01 to 2026-12-31." For expense entries, include a date. Gemini must be
reachable to interpret the question and call the expense tool.

If Gemini times out or returns an API error, no expense tool is called. Check
that the key in `.env` is active, has access to the configured model, and has
available quota. Do not share or paste the API key when troubleshooting.
