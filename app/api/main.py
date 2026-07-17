"""Optional REST API (M4) — thin wrapper over the same agent + ingestion code.

Run:  docker compose --profile api up   ->   http://localhost:8000/docs
"""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.agent.loop import run_agent
from app.ingestion.contracts import ingest_contract_bytes
from app.retrieval import store

app = FastAPI(title="Agent El-Dostor API", version="0.4.0")

DISCLAIMER = "General legal information, not a substitute for a licensed Egyptian lawyer."


class AskRequest(BaseModel):
    question: str
    contract_id: str | None = None
    history: list[dict[str, Any]] | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/contracts")
def list_contracts() -> dict[str, Any]:
    conn = store.connect(retries=1, delay=0.0)
    try:
        return {"contracts": store.list_contracts(conn)}
    finally:
        conn.close()


@app.post("/contracts")
async def upload_contract(
    file: UploadFile = File(...), contract_type: str = Form("unknown")
) -> dict[str, Any]:
    data = await file.read()
    try:
        return ingest_contract_bytes(file.filename or "upload", data, contract_type)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/ask")
def ask(req: AskRequest) -> dict[str, Any]:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question is required")
    try:
        result = run_agent(req.question, contract_id=req.contract_id, history=req.history)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    result["disclaimer"] = DISCLAIMER
    return result
