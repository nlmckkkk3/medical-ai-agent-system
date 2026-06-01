# MedChoice Implementation Plan

> **For agentic workers:** Use subagent-driven-development to implement task-by-task.

**Goal:** Build a medical consumer decision assistant with 4-agent architecture across 4 scenarios.

**Architecture:** 4 agents (Safety → Router → Business → Reflection) in pipeline, backed by ChromaDB RAG, document parser, and comparison engine. Streamlit Web UI.

**Tech Stack:** Python 3.10+, Streamlit, ChromaDB, sentence-transformers (BGE), PyMuPDF, unstructured, openai-compatible SDK for DeepSeek/智谱.

---

## File Structure

| File | Responsibility |
|------|---------------|
| `requirements.txt` | Python dependencies |
| `.env.example` | Environment variable template |
| `src/config.py` | Config loading from .env |
| `src/tools/llm_tool.py` | LLM API call wrapper |
| `src/tools/rag_tool.py` | ChromaDB vector store + retrieval |
| `src/tools/doc_parser.py` | PDF/image text extraction |
| `src/memory/user_memory.py` | User profile JSON persistence |
| `src/agents/safety_agent.py` | Input safety filter |
| `src/agents/router_agent.py` | Intent classification + routing |
| `src/agents/business_agent.py` | 4-scenario business logic orchestrator |
| `src/agents/reflection_agent.py` | 5-dimension output verification |
| `src/main.py` | Streamlit UI entry point |
| `src/init_kb.py` | Knowledge base initialization with sample data |
| `tests/test_cases.py` | 5 unit test cases |
| `tests/integration_test.py` | 6 integration test cases |
| `CLAUDE.md` | AI coding agent instructions |
| `README.md` | Project documentation |

## Dependency Order

```
requirements.txt → .env.example → config.py
                                    ↓
                              llm_tool.py
                              ↓
        ┌─────────────────────┼─────────────────────┐
        ↓                     ↓                     ↓
   rag_tool.py          doc_parser.py         user_memory.py
        │                     │
        └──────┬──────────────┘
               ↓
   ┌───────────┼───────────┐
   ↓           ↓           ↓
safety    router      reflection
   │         │            │
   └────┬────┘            │
        ↓                 │
   business_agent.py      │
        │                 │
        └─────────┬───────┘
                  ↓
              main.py
                  ↓
         init_kb.py, tests, CLAUDE.md, README.md
```

---
