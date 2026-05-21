# Enterprise AI Support Agent

An enterprise-grade AI support assistant built using RAG (Retrieval-Augmented Generation), LoRA fine-tuning, vector databases, hallucination evaluation, MCP-style tools, and conversational memory.

This project simulates an internal enterprise AI copilot capable of answering HR and IT support questions, retrieving information from enterprise documents, evaluating grounding quality, and performing support ticket operations.

---

# Features

## RAG Pipeline
- Semantic document retrieval
- ChromaDB vector database
- Embedding-based search
- Grounded AI responses

## Enterprise Document Support
- TXT document ingestion
- PDF upload support
- Dynamic enterprise knowledge base

## MCP-Style Tools
- Create support tickets
- Check ticket status
- Simulated enterprise tool orchestration

## Hallucination Evaluation
- Grounding validation
- Response-to-context comparison
- Basic hallucination detection

## Conversational Memory
- Multi-turn conversations
- Session-based chat history
- Chat reset functionality

## LoRA Fine-Tuning
- PEFT-based fine-tuning
- Lightweight domain adaptation
- Enterprise support instruction tuning

## Streamlit UI
- Interactive AI chat interface
- PDF upload interface
- Source visibility
- Enterprise support dashboard

---

# Architecture

```text
User
  ↓
Streamlit UI
  ↓
RAG Pipeline
  ↓
Chunking + Embeddings
  ↓
ChromaDB Vector Store
  ↓
LLM Response Generation
  ↓
Grounding Evaluation
  ↓
MCP-Style Support Tools

Tech Stack
AI / LLM
OpenAI GPT
Hugging Face Transformers
PEFT / LoRA
RAG
LangChain
ChromaDB
Sentence Transformers
Backend / UI
Streamlit
Python
Evaluation
Custom grounding evaluation


Project Strucute-
enterprise-ai-support-agent/
│
├── app/
│   └── main.py
│
├── src/
│   ├── rag_pipeline.py
│   ├── evaluation.py
│   ├── fine_tuning.py
│   └── test_lora.py
│
├── mcp_tools/
│   └── tools.py
│
├── data/
│   ├── hr_policy.txt
│   ├── it_support.txt
│   ├── security_policy.txt
│   └── support_finetune.jsonl
│
├── models/
│
├── chroma_db/
│
├── requirements.txt
├── README.md
├── .env
└── .gitignore


Enterprise Use Cases
Internal enterprise support assistant
HR policy assistant
IT helpdesk copilot
Employee handbook search
Enterprise document intelligence
AI-powered support automation


Future Improvements
FastAPI backend
LangGraph agent workflows
Real ticketing integrations
RAGAS evaluation
Authentication and RBAC
Docker deployment
Kubernetes deployment
Redis caching
Multi-user support