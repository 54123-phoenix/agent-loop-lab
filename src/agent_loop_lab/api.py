"""FastAPI boundary with validation and bounded session memory."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import inspect
import json
import os
from datetime import datetime
from typing import Any, Callable
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .agent import Agent, AgentRun
from .async_agent import AsyncAgent
from .coordination import (
    InMemoryRequestJournal,
    RequestJournal,
    SessionCoordinator,
    SQLiteRequestJournal,
)
from .context import ContextBudget, ContextManager
from .memory import (
    ConversationConflictError,
    ConversationSnapshot,
    ConversationStore,
    InMemoryConversationStore,
    SQLiteConversationStore,
)
from .long_term_memory import (
    InMemoryLongTermMemoryStore,
    LongTermMemoryStore,
    MemoryRecord,
    SQLiteLongTermMemoryStore,
)
from .models import Message
from .openai_model import AsyncOpenAIResponsesModel
from .tools import build_default_registry


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    session_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$",
    )
    message: str = Field(min_length=1, max_length=4000)
    reset: bool = False
    request_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$",
    )


class MessageView(BaseModel):
    role: str
    content: str
    name: str | None = None
    call_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    request_id: str
    run_id: str
    answer: str | None
    stop_reason: str
    steps: int
    retained_messages: int


class SessionResponse(BaseModel):
    session_id: str
    messages: list[MessageView]


class MemoryCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    category: str
    content: str = Field(min_length=1, max_length=2000)
    memory_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$",
    )
    source_event_ids: list[str] = Field(default_factory=list, max_length=20)
    confidence: float = Field(default=1.0, ge=0, le=1)
    expires_at: datetime | None = None


class MemoryView(BaseModel):
    memory_id: str
    session_id: str
    category: str
    content: str
    source_event_ids: list[str]
    confidence: float
    created_at: str
    expires_at: str | None


class MemoryListResponse(BaseModel):
    session_id: str
    memories: list[MemoryView]


AgentFactory = Callable[[], Agent | AsyncAgent]


def create_app(
    agent_factory: AgentFactory,
    *,
    store: ConversationStore | None = None,
    request_journal: RequestJournal | None = None,
    coordinator: SessionCoordinator | None = None,
    memory_store: LongTermMemoryStore | None = None,
) -> FastAPI:
    conversation_store = store if store is not None else InMemoryConversationStore()
    journal = request_journal or InMemoryRequestJournal()
    session_coordinator = coordinator or SessionCoordinator()
    long_term_store = memory_store or InMemoryLongTermMemoryStore()
    app = FastAPI(title="agent-loop-lab", version="0.10.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": "0.10.0"}

    @app.post("/v1/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        request_id = request.request_id or uuid4().hex
        fingerprint = _request_fingerprint(request)
        async with session_coordinator.hold(request.session_id):
            cached = journal.get(request.session_id, request_id)
            if cached is not None:
                if cached.get("request_fingerprint") != fingerprint:
                    raise HTTPException(
                        status_code=409,
                        detail={
                            "code": "idempotency_conflict",
                            "message": "request_id was already used with different input",
                        },
                    )
                return ChatResponse.model_validate(cached["response"])

            if request.reset:
                conversation_store.clear(request.session_id)
            snapshot = _load_snapshot(conversation_store, request.session_id)
            memories = long_term_store.search(
                request.session_id,
                request.message,
                limit=5,
            )
            try:
                pending_run: Any = agent_factory().run(
                    request.message,
                    history=snapshot.messages,
                    memories=tuple(memory.render() for memory in memories),
                )
                run: AgentRun = (
                    await pending_run if inspect.isawaitable(pending_run) else pending_run
                )
                _save_snapshot(
                    conversation_store,
                    request.session_id,
                    run.messages,
                    snapshot.version,
                )
            except ConversationConflictError as exc:
                raise HTTPException(
                    status_code=409,
                    detail={"code": "session_conflict", "message": str(exc)},
                ) from exc
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail={"code": "invalid_input", "message": str(exc)},
                ) from exc
            except Exception as exc:
                raise HTTPException(
                    status_code=502,
                    detail={
                        "code": "model_error",
                        "message": "The upstream model request failed",
                    },
                ) from exc
            retained = len(conversation_store.load(request.session_id))
            response = ChatResponse(
                session_id=request.session_id,
                request_id=request_id,
                run_id=run.run_id,
                answer=run.answer,
                stop_reason=run.stop_reason,
                steps=run.steps,
                retained_messages=retained,
            )
            journal.put(
                request.session_id,
                request_id,
                {
                    "request_fingerprint": fingerprint,
                    "response": response.model_dump(),
                },
            )
            return response

    @app.get("/v1/sessions/{session_id}", response_model=SessionResponse)
    def get_session(session_id: str) -> SessionResponse:
        try:
            messages = conversation_store.load(session_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail={"code": "invalid_session", "message": str(exc)}) from exc
        return SessionResponse(
            session_id=session_id,
            messages=[_message_view(message) for message in messages],
        )

    @app.delete("/v1/sessions/{session_id}")
    def delete_session(session_id: str) -> dict[str, object]:
        try:
            deleted = conversation_store.clear(session_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail={"code": "invalid_session", "message": str(exc)}) from exc
        return {"session_id": session_id, "deleted": deleted}

    @app.post(
        "/v1/sessions/{session_id}/memories",
        response_model=MemoryView,
        status_code=201,
    )
    def remember(session_id: str, request: MemoryCreateRequest) -> MemoryView:
        try:
            record = long_term_store.remember(
                MemoryRecord(
                    owner_id=session_id,
                    category=request.category,
                    content=request.content,
                    memory_id=request.memory_id or "",
                    source_event_ids=tuple(request.source_event_ids),
                    confidence=request.confidence,
                    expires_at=(
                        request.expires_at.isoformat() if request.expires_at else None
                    ),
                )
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail={"code": "invalid_memory", "message": str(exc)},
            ) from exc
        return _memory_view(record)

    @app.get(
        "/v1/sessions/{session_id}/memories",
        response_model=MemoryListResponse,
    )
    def list_memories(
        session_id: str,
        query: str = "",
        limit: int = 20,
    ) -> MemoryListResponse:
        try:
            records = (
                long_term_store.search(session_id, query, limit=limit)
                if query.strip()
                else long_term_store.list(session_id, limit=limit)
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail={"code": "invalid_memory_query", "message": str(exc)},
            ) from exc
        return MemoryListResponse(
            session_id=session_id,
            memories=[_memory_view(record) for record in records],
        )

    @app.delete("/v1/sessions/{session_id}/memories/{memory_id}")
    def forget_memory(session_id: str, memory_id: str) -> dict[str, object]:
        try:
            deleted = long_term_store.forget(session_id, memory_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail={"code": "invalid_memory", "message": str(exc)},
            ) from exc
        return {"session_id": session_id, "memory_id": memory_id, "deleted": deleted}

    return app


def _message_view(message: Message) -> MessageView:
    payload = asdict(message)
    return MessageView(
        role=payload["role"],
        content=payload["content"],
        name=payload["name"],
        call_id=payload["call_id"],
    )


def _memory_view(record: MemoryRecord) -> MemoryView:
    return MemoryView(
        memory_id=record.memory_id,
        session_id=record.owner_id,
        category=record.category,
        content=record.content,
        source_event_ids=list(record.source_event_ids),
        confidence=record.confidence,
        created_at=record.created_at,
        expires_at=record.expires_at,
    )


def _request_fingerprint(request: ChatRequest) -> str:
    payload = json.dumps(
        {"message": request.message, "reset": request.reset},
        ensure_ascii=False,
        sort_keys=True,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _load_snapshot(
    store: ConversationStore,
    session_id: str,
) -> ConversationSnapshot:
    loader = getattr(store, "load_snapshot", None)
    if loader is not None:
        return loader(session_id)
    return ConversationSnapshot(store.load(session_id), 0)


def _save_snapshot(
    store: ConversationStore,
    session_id: str,
    messages: tuple[Message, ...],
    expected_version: int,
) -> None:
    saver = getattr(store, "save_if_version", None)
    if saver is not None:
        saver(session_id, messages, expected_version=expected_version)
    else:
        store.save(session_id, messages)


def _default_agent_factory() -> AsyncAgent:
    return AsyncAgent(
        AsyncOpenAIResponsesModel.from_env(),
        build_default_registry(),
        context_manager=ContextManager(ContextBudget()),
    )


_database_path = os.getenv("AGENT_DB_PATH", "").strip()
if _database_path:
    _default_store: ConversationStore = SQLiteConversationStore(_database_path)
    _default_journal: RequestJournal = SQLiteRequestJournal(_database_path)
    _default_long_term_store: LongTermMemoryStore = SQLiteLongTermMemoryStore(
        _database_path
    )
else:
    _default_store = InMemoryConversationStore()
    _default_journal = InMemoryRequestJournal()
    _default_long_term_store = InMemoryLongTermMemoryStore()

app = create_app(
    _default_agent_factory,
    store=_default_store,
    request_journal=_default_journal,
    memory_store=_default_long_term_store,
)
