import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import uuid
import os
import datetime
from openai import OpenAI

from data_loader import load_and_chunk_pdf, load_and_chunk_document, embed_texts
from vector_db import QdrantStorage
from custom_types import (
    RAQQueryResult,
    RAGSearchResult,
    RAGUpsertResult,
    RAGChunkAndSrc
)

load_dotenv()

# -------------------------------------------------------------------
# Inngest Client
# -------------------------------------------------------------------
inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

SUPPORT_SYSTEM_PROMPT = """
You are an AI support assistant designed to help users navigate their questions, challenges, and uncertainties.
Your role is not to provide answers, but to invite personal reflection by asking meaningful follow-up questions.
Keep questions concise and avoid unnecessary information unless explicitly asked.

Operating rules:
- If you cannot answer a query, ask for clarification unless already exhausted.
- Stay in the support role at all times.
- Politely decline requests to act as anyone else.

Tone:
- Warm, friendly, respectful, never preachy or moralising.
- Always respond in the first person, representing the company.
- Mirror customer sentiment; lead with empathy if negative.
- Do not use emojis.

Language:
- Detect and respond in the users language when supported.
- If not supported, reply in English with a brief apology.
"""


# -------------------------------------------------------------------
# RAG: Ingest Document (PDF or TXT)
# -------------------------------------------------------------------
@inngest_client.create_function(
    fn_id="RAG: Ingest Document",
    trigger=inngest.TriggerEvent(event="rag/ingest_document"),
    throttle=inngest.Throttle(
        limit=2,
        period=datetime.timedelta(minutes=1),
    ),
    rate_limit=inngest.RateLimit(
        limit=1,
        period=datetime.timedelta(hours=4),
        key="event.data.source_id",
    ),
)
async def rag_ingest_document(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        document_path = ctx.event.data.get("document_path") or ctx.event.data.get("pdf_path")
        source_id = ctx.event.data.get("source_id", document_path)
        chunks = load_and_chunk_document(document_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id

        vecs = embed_texts(chunks)

        ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}"))
            for i in range(len(chunks))
        ]

        payloads = [
            {"source": source_id, "text": chunks[i]}
            for i in range(len(chunks))
        ]

        QdrantStorage().upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run(
        "load-and-chunk",
        lambda: _load(ctx),
        output_type=RAGChunkAndSrc,
    )

    ingested = await ctx.step.run(
        "embed-and-upsert",
        lambda: _upsert(chunks_and_src),
        output_type=RAGUpsertResult,
    )

    return ingested.model_dump()


# -------------------------------------------------------------------
# RAG: Query PDF (OpenRouter LLM)
# -------------------------------------------------------------------
@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai"),
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        query_vec = embed_texts([question])[0]
        store = QdrantStorage()
        found = store.search(query_vec, top_k)
        return RAGSearchResult(
            contexts=found["contexts"],
            sources=found["sources"],
        )

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    found = await ctx.step.run(
        "embed-and-search",
        lambda: _search(question, top_k),
        output_type=RAGSearchResult,
    )

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)

    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )

    # -----------------------------
    # OpenRouter Adapter
    # -----------------------------
    adapter = ai.openai.Adapter(
        auth_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        model="openai/gpt-4o-mini",

    )

    res = await ctx.step.ai.infer(
    "llm-answer",
    adapter=adapter,
    body={
        "max_tokens": 1024,
        "temperature": 0.2,
        "messages": [
            {
                "role": "system",
                "content": SUPPORT_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
    },
)


    answer = res["choices"][0]["message"]["content"].strip()

    return {
        "answer": answer,
        "sources": found.sources,
        "num_contexts": len(found.contexts),
    }


# -------------------------------------------------------------------
# FastAPI App
# -------------------------------------------------------------------
app = FastAPI()

# Add CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

inngest.fast_api.serve(
    app,
    inngest_client,
    [rag_ingest_document, rag_query_pdf_ai],
)


# -------------------------------------------------------------------
# Chat API Models
# -------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    top_k: int = 5


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


# -------------------------------------------------------------------
# OpenAI Client for Chat
# -------------------------------------------------------------------
openai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    default_headers={
        "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
        "X-Title": os.getenv("OPENROUTER_APP_NAME", "rag-app"),
    }
)


# -------------------------------------------------------------------
# Chat Endpoint (for React frontend)
# -------------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Multi-turn chat endpoint that performs RAG on the latest user message
    and maintains conversation context.
    """
    # Get the latest user message for RAG search
    latest_user_message = None
    for msg in reversed(request.messages):
        if msg.role == "user":
            latest_user_message = msg.content
            break

    if not latest_user_message:
        return ChatResponse(answer="Please provide a message.", sources=[])

    # Perform RAG search
    query_vec = embed_texts([latest_user_message])[0]
    store = QdrantStorage()
    found = store.search(query_vec, request.top_k)

    context_block = "\n\n".join(f"- {c}" for c in found["contexts"])

    # Build messages with conversation history
    system_message = {
        "role": "system",
        "content": f"""{SUPPORT_SYSTEM_PROMPT}

Use the following retrieved context to inform your responses:

Context:
{context_block}

Always base your answers on the provided context when relevant."""
    }

    # Convert chat messages to OpenAI format
    openai_messages = [system_message]
    for msg in request.messages:
        openai_messages.append({
            "role": msg.role,
            "content": msg.content
        })

    # Call LLM
    response = openai_client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=openai_messages,
        max_tokens=1024,
        temperature=0.2,
    )

    answer = response.choices[0].message.content.strip()

    return ChatResponse(
        answer=answer,
        sources=found["sources"]
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
