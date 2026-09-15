"""FastAPI boundary with validation and bounded session memory."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import inspect
import json
import os
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


AgentFactory = Callable[[], Agent | AsyncAgent]


def create_app(
    agent_factory: AgentFactory,
    *,
    store: ConversationStore | None = None,
    request_journal: RequestJournal | None = None,
    coordinator: SessionCoordinator | None = None,
) -> FastAPI:
    conversation_store = store if store is not None else InMemoryConversationStore()
    journal = request_journal or InMemoryRequestJournal()
    session_coordinator = coordinator or SessionCoordinator()
    app = FastAPI(title="agent-loop-lab", version="0.9.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": "0.9.0"}

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
            try:
                pending_run: Any = agent_factory().run(
                    request.message,
                    history=snapshot.messages,
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

    return app


def _message_view(message: Message) -> MessageView:
    payload = asdict(message)
    return MessageView(
        role=payload["role"],
        content=payload["content"],
        name=payload["name"],
        call_id=payload["call_id"],
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
else:
    _default_store = InMemoryConversationStore()
    _default_journal = InMemoryRequestJournal()

app = create_app(
    _default_agent_factory,
    store=_default_store,
    request_journal=_default_journal,
)
