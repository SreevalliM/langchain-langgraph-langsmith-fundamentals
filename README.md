# LangChain, LangGraph & LangSmith Fundamentals

A hands-on learning path covering LLM application development with LangChain, graph-based agent workflows with LangGraph, and evaluation/tracing with LangSmith.

## Setup

```bash
python3.12 -m venv lc-venv
source lc-venv/bin/activate
pip install -r requirements.txt
```

You will need API keys for the LLM providers used throughout the notebooks (OpenAI, Anthropic, Google, Groq, etc.). Set them as environment variables or in a `.env` file.

## Repository Structure

### [`langchain-agents/`](langchain-agents/)

Nine progressive lessons on building agents with LangChain:

| Lesson | Topic |
|--------|-------|
| L1 | Fast agent setup |
| L2 | Message handling |
| L3 | Streaming responses |
| L4 | Tool integration |
| L5 | Tools with MCP |
| L6 | Conversation memory |
| L7 | Structured output |
| L8 | Dynamic agent behavior |
| L9 | Human-in-the-Loop |

### [`langchain-foundations/`](langchain-foundations/)

**Module 1 — Foundational Concepts**
- Model initialization, invocation, and provider comparison (GPT, Claude, Gemini)
- Prompt engineering techniques
- Tool use and web search
- Conversation memory
- Multimodal messages (images, audio)
- Capstone: Personal Chef agent

**Module 2 — Multi-Agent Systems & RAG**
- Model Context Protocol (MCP) server integration
- Runtime context and state management
- Multi-agent coordination
- Capstone: Wedding Planner multi-agent system
- Bonus: Retrieval-Augmented Generation (RAG) and SQL agents

**Module 3 — Advanced Patterns**
- Message trimming, filtering, and optimization
- Human-in-the-Loop approval workflows
- Dynamic models, prompts, and tools at runtime
- Capstone: Email agent with Chat UI

### [`langgraph-foundations/`](langgraph-foundations/)

**Module 0 — LangGraph Basics**
- Nodes, edges, conditional routing
- Memory and interrupts
- Email agent example

**Module 1 — Building Agents**
- ReAct pattern (Act → Observe → Reason)
- Chains, routers, and simple graphs
- Agent memory and deployment

**Module 2 — State Management**
- State schemas and TypedDict definitions
- Reducers for state updates
- External memory, summarization, and message trimming
- Multiple state schemas

**Module 3 — Human-in-the-Loop**
- Breakpoints for pausing/resuming execution
- Editing state with human feedback
- Dynamic (conditional) breakpoints

**Module 4 — Advanced Patterns**
- Map-reduce for parallel processing

### [`langsmith/`](langsmith/)

LangSmith tracing and evaluation:
- Prompt optimization with ELI5 examples
- Experiment tracking and evaluation datasets
- Full explain-bot example with LangGraph Studio config

### [`langsmith-cookbook/`](langsmith-cookbook/)

**1 — Introduction**: Getting started with LangSmith tracing and evaluation.

**2 — Optimization**: Assisted prompt engineering, few-shot bootstrapping, systematic optimization workflows, fine-tuning on chat runs, and Lilac data exploration.

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `langchain` / `langchain-core` | LLM framework |
| `langgraph` / `langgraph-prebuilt` | Graph-based agent workflows |
| `langsmith` | Tracing and evaluation |
| `langchain-openai`, `-groq`, `-ollama`, `-huggingface` | LLM providers |
| `langchain-mcp-adapters` | Model Context Protocol |
| `tavily-python`, `wikipedia`, `arxiv` | Search and knowledge tools |
| `chromadb` | Vector database (RAG) |
| `pypdf` | PDF processing |
