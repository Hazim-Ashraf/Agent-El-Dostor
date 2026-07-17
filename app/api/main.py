"""Optional REST API (M4) — thin wrapper over the same agent + ingestion code.

Run:  docker compose --profile api up   ->   http://localhost:8000/docs
"""
from __future__ import annotations

import io
import zipfile
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from pydantic import BaseModel

from app.agent.loop import run_agent
from app.generation.loop import generate_contract
from app.generation.pdf import render_contract_pdfs
from app.ingestion.contracts import ingest_contract_bytes
from app.retrieval import store

app = FastAPI(title="Agent El-Dostor API", version="0.5.0")

DISCLAIMER = "General legal information, not a substitute for a licensed Egyptian lawyer."


class AskRequest(BaseModel):
    question: str
    contract_id: str | None = None
    history: list[dict[str, Any]] | None = None


class GenerateRequest(BaseModel):
    contract_type: str
    language: str = "bilingual"          # ar | en | bilingual
    brief: str = ""


class GeneratePdfRequest(GenerateRequest):
    layout: str = "side_by_side"         # side_by_side | sequential | separate (bilingual only)


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


@app.post("/generate")
def generate(req: GenerateRequest) -> dict[str, Any]:
    """Draft a contract; returns the structured contract + citation verification."""
    try:
        result = generate_contract(req.contract_type, language=req.language, brief=req.brief)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    result["disclaimer"] = DISCLAIMER
    return result


@app.post("/generate/pdf")
def generate_pdf(req: GeneratePdfRequest) -> Response:
    """Draft a contract and return it as a PDF (a zip when layout='separate')."""
    try:
        result = generate_contract(req.contract_type, language=req.language, brief=req.brief)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if result.get("status") != "generated" or not result.get("contract"):
        raise HTTPException(status_code=502, detail=result.get("error") or "no contract produced")

    try:
        pdfs = render_contract_pdfs(result["contract"], req.language, req.layout)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"PDF rendering failed: {exc}") from exc

    if len(pdfs) == 1:
        name, data = pdfs[0]
        return Response(
            content=data,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{name}"'},
        )
    # multiple files (separate) -> zip
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in pdfs:
            zf.writestr(name, data)
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="contract_pdfs.zip"'},
    )
