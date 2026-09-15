"""FastAPI boundary with validation and bounded session memory."""

from __future__ import annotations

from dataclasses import asdict
import inspect
from typing import Any, Callable

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .agent import Agent, AgentRun
from .async_agent import AsyncAgent
from .memory import ConversationStore, InMemoryConversationStore
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


class MessageView(BaseModel):
    role: str
    content: str
    name: str | None = None
    call_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
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
) -> FastAPI:
    conversation_store = store if store is not None else InMemoryConversationStore()
    app = FastAPI(title="agent-loop-lab", version="0.5.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": "0.5.0"}

    @app.post("/v1/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        if request.reset:
            conversation_store.clear(request.session_id)
        history = conversation_store.load(request.session_id)
        try:
            pending_run: Any = agent_factory().run(request.message, history=history)
            run: AgentRun = (
                await pending_run if inspect.isawaitable(pending_run) else pending_run
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail={"code": "invalid_input", "message": str(exc)}) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail={
                    "code": "model_error",
                    "message": "The upstream model request failed",
                },
            ) from exc
        conversation_store.save(request.session_id, run.messages)
        retained = len(conversation_store.load(request.session_id))
        return ChatResponse(
            session_id=request.session_id,
            run_id=run.run_id,
            answer=run.answer,
            stop_reason=run.stop_reason,
            steps=run.steps,
            retained_messages=retained,
        )

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


def _default_agent_factory() -> AsyncAgent:
    return AsyncAgent(
        AsyncOpenAIResponsesModel.from_env(),
        build_default_registry(),
    )


app = create_app(_default_agent_factory)
