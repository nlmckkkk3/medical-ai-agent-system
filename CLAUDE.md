# CLAUDE.md

This file provides guidance to Claude Code when working with this project.

## Project Overview
MedChoice is a medical consumer decision assistant — an AI agent application that helps users make informed decisions about physical exam packages, drug comparisons, health report interpretation, and medical department recommendations. It uses a 4-agent pipeline (Safety → Router → Business → Reflection) with ChromaDB RAG (BGE embeddings), streaming LLM output, and Streamlit UI.

## Architecture
```
User Input → Safety Agent → Router Agent → Business Agent → Reflection Agent → Output
```

## Running the App
```bash
# Install dependencies
pip install -r requirements.txt

# Set up API keys
cp .env.example .env
# Edit .env with your DeepSeek or Zhipu API key

# Initialize knowledge base (first time only)
python src/init_kb.py

# Run the app
streamlit run src/main.py
```

## Running Tests
```bash
python tests/test_cases.py
```

## Key Files
- `src/main.py` — Streamlit UI entry point
- `src/agents/safety_agent.py` — Input safety filter
- `src/agents/router_agent.py` — Intent classification (4 scenarios)
- `src/agents/business_agent.py` — Core business logic orchestrator
- `src/agents/reflection_agent.py` — Output quality verification (5 dimensions)
- `src/tools/rag_tool.py` — ChromaDB vector search
- `src/tools/doc_parser.py` — PDF/text document parsing
- `src/tools/llm_tool.py` — LLM API wrapper (DeepSeek/Zhipu)
- `src/memory/user_memory.py` — User profile JSON persistence
- `src/init_kb.py` — Knowledge base initialization with sample data
- `tests/test_cases.py` — 5 unit test cases
- `tests/integration_test.py` — 6 end-to-end integration tests (requires API key)
- `.streamlit/config.toml` — Dark glass-morphism theme configuration
- `src/utils.py` — Shared JSON parser used by 3 agent modules
- `src/app.py` — Gradio secondary interface

## Code Style
- Python 3.10+
- Type hints on function signatures
- Chinese content for user-facing strings, English for code identifiers
- Each agent has a SYSTEM prompt constant at module level
- No real API keys in code — always from environment variables
