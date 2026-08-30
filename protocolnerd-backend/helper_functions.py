from __future__ import annotations

import json
import logging
import os
import re
import html
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ollama_executions import _retryable_ollama_call

logging.basicConfig(level=logging.INFO)

# Load environment variables once on import.
load_dotenv("variables.env", override=True)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 8
DEFAULT_EXECUTION_STRATEGY = "agentic"

LOCAL_ANALYSIS_SYSTEM_PROMPT = """
You are a careful legal/compliance analysis assistant running entirely locally.

Your job:
1. Read the target document.
2. Read the retrieved context passages from source documents.
3. Determine whether the target document appears consistent with, inconsistent with,
   or under-specified relative to the retrieved context.
4. Be explicit about uncertainty.
5. Ground every finding in quoted evidence from the retrieved context and the target document.
6. Do not invent laws, sections, or requirements not present in the provided materials.
7. If the provided context is insufficient, say so clearly.

Return valid JSON with this exact schema:
{
  "summary": "short overall conclusion",
  "overall_risk": "low|medium|high|unclear",
  "findings": [
    {
      "issue": "short issue label",
      "risk_level": "low|medium|high|unclear",
      "status": "compliant|non_compliant|needs_review|unclear",
      "explanation": "plain-English explanation",
      "target_evidence": ["quote or excerpt from target"],
      "context_evidence": [
        {
          "source_file": "filename",
          "chunk_id": "chunk identifier",
          "quote": "supporting quote"
        }
      ],
      "recommendation": "specific next step"
    }
  ],
  "gaps": ["missing info or ambiguity 1"],
  "notes": ["additional grounded note 1"]
}
""".strip()

CUSTOM_PROMPT_SYSTEM_PROMPT = """
You are a document analysis assistant running entirely locally.

You will receive:
- A specific check to perform (the user's prompt)
- The target document text
- Retrieved context passages from governing documents

Your job: evaluate the target document against the specific check provided.
Be explicit about uncertainty. Quote evidence from both the target and the context.
Do not invent requirements not present in the supplied materials.

Return valid JSON with this exact schema:
{
  "issue": "short label summarizing what was checked",
  "status": "compliant|non_compliant|unclear",
  "risk_level": "low|medium|high|unclear",
  "explanation": "2-4 sentence plain-English explanation of your finding",
  "recommendation": "specific next step if action is needed, or empty string",
  "target_evidence": ["relevant quote or excerpt from the target document"],
  "context_evidence": [
    {
      "source_file": "filename",
      "chunk_id": "chunk identifier",
      "quote": "supporting quote from context passage"
    }
  ]
}
""".strip()

PROMPT_BASED_ANALYSIS_SYSTEM_PROMPT = """
You are a careful legal/compliance analysis assistant running entirely locally.

You will receive:
- the user's analysis request
- the target document text
- retrieved context passages from governing documents

Your job:
1. Analyze the target against the retrieved context in a single pass.
2. Be explicit about uncertainty and missing information.
3. Ground every finding in quoted evidence from both the target and the context.
4. Do not invent requirements not present in the supplied materials.
5. Return valid JSON only.

Return JSON with this exact schema:
{
  "summary": "3-5 sentence summary of the target document",
  "synthesis": "final report with overall verdict, key issues, and recommendations",
  "overall_risk": "low|medium|high|unclear",
  "status_counts": {
    "compliant": 0,
    "non_compliant": 0,
    "unclear": 0
  },
  "findings": [
    {
      "issue": "short issue label",
      "status": "compliant|non_compliant|unclear",
      "risk_level": "low|medium|high|unclear",
      "explanation": "plain-English explanation",
      "recommendation": "specific next step",
      "target_evidence": ["quote or excerpt from target"],
      "context_evidence": [
        {
          "source_file": "filename",
          "chunk_id": "chunk identifier",
          "quote": "supporting quote"
        }
      ]
    }
  ],
  "gaps": ["missing info or ambiguity 1"],
  "notes": ["additional grounded note 1"]
}
""".strip()


# ---------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------

def ensure_session_dirs(base_dir: Path, session_id: str) -> Dict[str, Path]:
    """
    Create and return the directory structure for a processing session.

    Returns a dictionary with:
    - root
    - context
    - target
    - extracted
    - index
    - outputs
    """
    root = Path(base_dir) / session_id
    dirs = {
        "root": root,
        "context": root / "context",
        "target": root / "target",
        "extracted": root / "extracted",
        "index": root / "index",
        "outputs": root / "outputs",
    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


# ---------------------------------------------------------------------
# Text cleaning / extraction
# ---------------------------------------------------------------------

def clean_text(raw: Any) -> str:
    """
    Normalize raw text or HTML-ish content into readable plain text.
    """
    if raw is None:
        return ""

    if not isinstance(raw, str):
        try:
            raw = json.dumps(raw, ensure_ascii=False)
        except Exception:
            raw = str(raw)

    if not raw:
        return ""

    # Decode escaped unicode where possible.
    try:
        raw = raw.encode("utf-8").decode("unicode_escape", errors="ignore")
    except Exception:
        pass

    text = html.unescape(raw)

    # Strip HTML tags if present.
    if "<" in text and ">" in text:
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text(separator="\n")

    # Normalize line endings and whitespace.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove repeated non-content artifacts.
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\[\d+\]", "", text)

    return text.strip()


def extract_text_from_pdf(path: Path) -> str:
    """
    Extract text from a PDF using PyMuPDF.
    """
    text_parts: List[str] = []

    with fitz.open(path) as doc:
        for page in doc:
            page_text = page.get_text("text") or ""
            if page_text.strip():
                text_parts.append(page_text)

    return clean_text("\n\n".join(text_parts))


def extract_text_from_txt(path: Path) -> str:
    """
    Extract text from plain text / markdown / code-ish files.
    """
    raw = path.read_text(encoding="utf-8", errors="ignore")
    return clean_text(raw)


def extract_text_from_upload(path: Path) -> str:
    """
    Extract text from an uploaded file path.

    Supported:
    - pdf
    - txt
    - md
    - html
    - htm
    - json
    - csv
    - py
    - js
    - ts
    - yaml
    - yml
    - xml

    Falls back to UTF-8 text read for unknown text-like files.
    """
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return extract_text_from_pdf(path)

    text_like_suffixes = {
        ".txt", ".md", ".html", ".htm", ".json", ".csv",
        ".py", ".js", ".ts", ".yaml", ".yml", ".xml", ".log"
    }

    if suffix in text_like_suffixes:
        return extract_text_from_txt(path)

    # Generic fallback
    try:
        return extract_text_from_txt(path)
    except Exception as exc:
        raise ValueError(f"Unsupported or unreadable file type for {path.name}: {exc}") from exc


# ---------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------

def split_text_into_chunks(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into overlapping character-based chunks.

    This is intentionally simple and local-only. For legal/compliance corpora,
    preserving nearby context matters, so chunks overlap.
    """
    text = (text or "").strip()
    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: List[str] = []
    start = 0
    step = chunk_size - chunk_overlap

    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += step

    return chunks


def build_chunk_records(
    context_documents: List[Dict[str, Any]],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Dict[str, Any]]:
    """
    Convert full documents into chunk records with metadata.
    """
    records: List[Dict[str, Any]] = []

    for doc in context_documents:
        source_file = doc.get("source_file", "unknown")
        text = doc.get("text", "") or ""
        path = doc.get("path", "")

        chunks = split_text_into_chunks(
            text=text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for idx, chunk in enumerate(chunks):
            records.append({
                "chunk_id": f"{source_file}::chunk_{idx + 1}",
                "source_file": source_file,
                "path": path,
                "text": chunk,
                "chunk_index": idx + 1,
            })

    return records


# ---------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------

def build_local_rag_index(
    context_documents: List[Dict[str, Any]],
    index_dir: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Dict[str, Any]:
    """
    Build a local TF-IDF retrieval index over context document chunks.

    Returns a serializable dict containing:
    - chunk records
    - vectorizer
    - matrix

    Also writes chunk metadata to disk for inspection.
    """
    index_dir.mkdir(parents=True, exist_ok=True)

    chunk_records = build_chunk_records(
        context_documents=context_documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if not chunk_records:
        raise ValueError("No context chunks were created. Check the uploaded context files.")

    corpus = [record["text"] for record in chunk_records]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=50000,
    )
    matrix = vectorizer.fit_transform(corpus)

    metadata_path = index_dir / "chunk_records.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(chunk_records, f, indent=2, ensure_ascii=False)

    return {
        "chunk_records": chunk_records,
        "vectorizer": vectorizer,
        "matrix": matrix,
        "metadata_path": str(metadata_path),
    }


def build_retrieval_query(user_query: str, target_text: str, max_target_chars: int = 2500) -> str:
    """
    Combine the explicit user query with a clipped prefix of the target text to help retrieval.
    """
    target_excerpt = (target_text or "")[:max_target_chars]
    parts = [
        user_query.strip() if user_query else "",
        target_excerpt.strip(),
    ]
    return "\n\n".join([p for p in parts if p]).strip()


def retrieve_relevant_chunks(
    rag_index: Dict[str, Any],
    user_query: str,
    target_text: str,
    top_k: int = DEFAULT_TOP_K,
) -> List[Dict[str, Any]]:
    """
    Retrieve the most relevant context chunks using cosine similarity over TF-IDF vectors.
    """
    chunk_records = rag_index["chunk_records"]
    vectorizer = rag_index["vectorizer"]
    matrix = rag_index["matrix"]

    retrieval_query = build_retrieval_query(user_query=user_query, target_text=target_text)
    if not retrieval_query.strip():
        raise ValueError("Retrieval query is empty.")

    query_vec = vectorizer.transform([retrieval_query])
    scores = cosine_similarity(query_vec, matrix).flatten()

    ranked_indices = scores.argsort()[::-1][:max(top_k, 1)]

    results: List[Dict[str, Any]] = []
    for idx in ranked_indices:
        record = chunk_records[idx].copy()
        record["score"] = float(scores[idx])
        results.append(record)

    return results


# ---------------------------------------------------------------------
# Ollama analysis
# ---------------------------------------------------------------------

def _safe_json_loads(raw: str) -> Dict[str, Any]:
    """
    Parse possibly fenced or slightly malformed JSON.
    """
    if not raw:
        return {}

    cleaned = re.sub(r"```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```", "", cleaned)
    cleaned = cleaned.strip()
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logging.warning("Failed to parse model JSON output.")
        return {}


def truncate_text(text: str, max_chars: int) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated]"


def get_default_execution_strategy() -> str:
    strategy = (os.getenv("EXECUTION_STRATEGY", DEFAULT_EXECUTION_STRATEGY) or "").strip().lower()
    if strategy in {"prompt", "prompt_based", "single_prompt"}:
        return "prompt_based"
    return DEFAULT_EXECUTION_STRATEGY


def normalize_execution_strategy(value: Optional[str]) -> str:
    normalized = (value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "agentic": "agentic",
        "multi_step": "agentic",
        "pipeline": "agentic",
        "prompt": "prompt_based",
        "prompt_based": "prompt_based",
        "single_prompt": "prompt_based",
    }
    return aliases.get(normalized, get_default_execution_strategy())


def format_retrieved_chunks_for_prompt(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Render retrieved chunks into a prompt-friendly block.
    """
    formatted: List[str] = []

    for chunk in retrieved_chunks:
        formatted.append(
            "\n".join([
                f"Source File: {chunk.get('source_file', '')}",
                f"Chunk ID: {chunk.get('chunk_id', '')}",
                f"Similarity Score: {chunk.get('score', 0.0):.4f}",
                "Context Passage:",
                chunk.get("text", ""),
            ])
        )

    return "\n\n---\n\n".join(formatted)


def _normalize_prompt_based_result(parsed: Dict[str, Any]) -> Dict[str, Any]:
    findings = parsed.get("findings")
    if not isinstance(findings, list):
        findings = []

    normalized_findings: List[Dict[str, Any]] = []
    compliant_count = 0
    non_compliant_count = 0
    unclear_count = 0

    for finding in findings:
        if not isinstance(finding, dict):
            continue

        status = str(finding.get("status", "unclear")).strip().lower()
        if status not in {"compliant", "non_compliant", "unclear"}:
            if status in {"non-compliant", "noncompliant"}:
                status = "non_compliant"
            else:
                status = "unclear"

        if status == "compliant":
            compliant_count += 1
        elif status == "non_compliant":
            non_compliant_count += 1
        else:
            unclear_count += 1

        context_evidence = finding.get("context_evidence")
        if not isinstance(context_evidence, list):
            context_evidence = []

        normalized_findings.append({
            "issue": finding.get("issue", "Untitled finding"),
            "status": status,
            "risk_level": finding.get("risk_level", "unclear"),
            "explanation": finding.get("explanation", ""),
            "recommendation": finding.get("recommendation", ""),
            "target_evidence": finding.get("target_evidence", []),
            "context_evidence": [
                {
                    "source_file": item.get("source_file", ""),
                    "chunk_id": item.get("chunk_id", ""),
                    "quote": item.get("quote", ""),
                }
                for item in context_evidence
                if isinstance(item, dict)
            ],
        })

    status_counts = parsed.get("status_counts")
    if not isinstance(status_counts, dict):
        status_counts = {
            "compliant": compliant_count,
            "non_compliant": non_compliant_count,
            "unclear": unclear_count,
        }

    return {
        "summary": parsed.get("summary", ""),
        "synthesis": parsed.get("synthesis", ""),
        "overall_risk": parsed.get("overall_risk", "unclear"),
        "status_counts": {
            "compliant": _safe_int(status_counts.get("compliant", compliant_count), compliant_count),
            "non_compliant": _safe_int(status_counts.get("non_compliant", non_compliant_count), non_compliant_count),
            "unclear": _safe_int(status_counts.get("unclear", unclear_count), unclear_count),
        },
        "findings": normalized_findings,
        "gaps": parsed.get("gaps", []) if isinstance(parsed.get("gaps"), list) else [],
        "notes": parsed.get("notes", []) if isinstance(parsed.get("notes"), list) else [],
    }


def _agent_summarize_target(target_text: str) -> str:
    """Agent step 1: Produce a concise summary of the target document."""
    excerpt = truncate_text(target_text, 6000)
    raw = _retryable_ollama_call(
        messages=[
            {"role": "system", "content": (
                "You are a document summarizer. Read the document below and write a concise "
                "summary (3-5 sentences). Focus on: what the document is, who the parties are, "
                "what it proposes or requires, and any key terms or deadlines. "
                "Do NOT add information that is not in the document."
            )},
            {"role": "user", "content": excerpt},
        ],
        temperature=0.1,
    )
    return raw.strip() if raw else "Could not summarize the target document."


def _agent_evaluate_chunk(
    user_query: str,
    target_summary: str,
    target_excerpt: str,
    chunk: Dict[str, Any],
) -> Dict[str, Any]:
    """Agent step 2: Evaluate the target against one context chunk."""
    raw = _retryable_ollama_call(
        messages=[
            {"role": "system", "content": (
                "You are a compliance analyst. You will receive:\n"
                "- A user question\n"
                "- A summary of a target document\n"
                "- A short excerpt from the target document\n"
                "- A passage from a governing/context document\n\n"
                "Your job: Does the target document satisfy the requirements in the context passage?\n\n"
                "Reply with EXACTLY this format (plain text, not JSON):\n"
                "STATUS: compliant | non_compliant | unclear\n"
                "ISSUE: one sentence describing the finding\n"
                "EVIDENCE: quote a short phrase from the context passage that supports your finding\n"
                "EXPLANATION: 2-3 sentences explaining your reasoning\n"
            )},
            {"role": "user", "content": (
                f"USER QUESTION: {user_query}\n\n"
                f"TARGET SUMMARY: {target_summary}\n\n"
                f"TARGET EXCERPT:\n{truncate_text(target_excerpt, 2000)}\n\n"
                f"CONTEXT PASSAGE (from {chunk.get('source_file', 'unknown')}):\n"
                f"{chunk.get('text', '')}"
            )},
        ],
        temperature=0.1,
    )

    # Parse the structured text response
    finding = {
        "source_file": chunk.get("source_file", ""),
        "chunk_id": chunk.get("chunk_id", ""),
        "status": "unclear",
        "issue": "",
        "evidence": "",
        "explanation": "",
        "raw": raw.strip() if raw else "",
    }

    if raw:
        for line in raw.strip().split("\n"):
            line_upper = line.strip().upper()
            value = line.split(":", 1)[1].strip() if ":" in line else ""
            if line_upper.startswith("STATUS:"):
                status_val = value.lower().strip()
                if "non_compliant" in status_val or "non-compliant" in status_val:
                    finding["status"] = "non_compliant"
                elif "compliant" in status_val:
                    finding["status"] = "compliant"
                else:
                    finding["status"] = "unclear"
            elif line_upper.startswith("ISSUE:"):
                finding["issue"] = value
            elif line_upper.startswith("EVIDENCE:"):
                finding["evidence"] = value
            elif line_upper.startswith("EXPLANATION:"):
                finding["explanation"] = value

    return finding


def _agent_synthesize(
    user_query: str,
    target_summary: str,
    findings: List[Dict[str, Any]],
) -> str:
    """Agent step 3: Synthesize all chunk findings into a final verdict."""
    findings_text = ""
    for i, f in enumerate(findings, 1):
        findings_text += (
            f"Finding {i} ({f['source_file']}):\n"
            f"  Status: {f['status']}\n"
            f"  Issue: {f['issue']}\n"
            f"  Explanation: {f['explanation']}\n\n"
        )

    raw = _retryable_ollama_call(
        messages=[
            {"role": "system", "content": (
                "You are a compliance report writer. You will receive a target document summary "
                "and a list of individual findings from comparing it against governing documents.\n\n"
                "Write a clear final report with:\n"
                "1. OVERALL VERDICT: Is the target document compliant, non-compliant, or unclear? One sentence.\n"
                "2. KEY ISSUES: List the most important problems found (if any). Be specific.\n"
                "3. RECOMMENDATIONS: What should be done next?\n\n"
                "Be direct and concise. Do not repeat the full findings — summarize them."
            )},
            {"role": "user", "content": (
                f"USER QUESTION: {user_query}\n\n"
                f"TARGET DOCUMENT SUMMARY:\n{target_summary}\n\n"
                f"INDIVIDUAL FINDINGS:\n{findings_text}"
            )},
        ],
        temperature=0.2,
    )
    return raw.strip() if raw else "Could not generate final synthesis."


def _run_single_custom_prompt(
    custom_prompt: str,
    target_text: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run one user-supplied prompt against the target document and context."""
    target_excerpt = truncate_text(target_text, 8000)
    context_block = format_retrieved_chunks_for_prompt(retrieved_chunks[:8])

    raw = _retryable_ollama_call(
        messages=[
            {"role": "system", "content": CUSTOM_PROMPT_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"CHECK TO PERFORM:\n{custom_prompt}\n\n"
                f"TARGET DOCUMENT:\n{target_excerpt}\n\n"
                f"RETRIEVED CONTEXT PASSAGES:\n{context_block}"
            )},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    parsed = _safe_json_loads(raw)
    if not parsed:
        return {
            "issue": custom_prompt[:100],
            "status": "unclear",
            "risk_level": "unclear",
            "explanation": "Could not parse a structured response for this prompt.",
            "recommendation": "",
            "target_evidence": [],
            "context_evidence": [],
        }

    status = str(parsed.get("status", "unclear")).strip().lower()
    if status not in {"compliant", "non_compliant", "unclear"}:
        status = "non_compliant" if "non" in status else "unclear"

    context_evidence = parsed.get("context_evidence", [])
    if not isinstance(context_evidence, list):
        context_evidence = []

    return {
        "issue": parsed.get("issue") or custom_prompt[:100],
        "status": status,
        "risk_level": parsed.get("risk_level", "unclear"),
        "explanation": parsed.get("explanation", ""),
        "recommendation": parsed.get("recommendation", ""),
        "target_evidence": parsed.get("target_evidence", []) if isinstance(parsed.get("target_evidence"), list) else [],
        "context_evidence": [
            {
                "source_file": item.get("source_file", ""),
                "chunk_id": item.get("chunk_id", ""),
                "quote": item.get("quote", ""),
            }
            for item in context_evidence
            if isinstance(item, dict)
        ],
    }


def _prompt_based_analyze_target(
    user_query: str,
    target_text: str,
    retrieved_chunks: List[Dict[str, Any]],
    custom_prompts: Optional[List[str]] = None,
    on_progress=None,
) -> Dict[str, Any]:
    def _progress(msg):
        if on_progress:
            on_progress(msg)

    if custom_prompts:
        findings = []
        total = len(custom_prompts)
        for i, prompt in enumerate(custom_prompts, 1):
            _progress(f"Running custom prompt {i}/{total}: {prompt[:70]}...")
            finding = _run_single_custom_prompt(
                custom_prompt=prompt,
                target_text=target_text,
                retrieved_chunks=retrieved_chunks,
            )
            findings.append(finding)
            _progress(f"  Prompt {i} result: {finding['status']} — {finding['issue'][:60]}")

        compliant = sum(1 for f in findings if f["status"] == "compliant")
        non_compliant = sum(1 for f in findings if f["status"] == "non_compliant")
        unclear = sum(1 for f in findings if f["status"] == "unclear")

        if non_compliant > 0:
            overall_risk = "high"
        elif unclear > compliant:
            overall_risk = "medium"
        else:
            overall_risk = "low"

        issue_labels = [f["issue"] for f in findings if f["status"] == "non_compliant"]
        synthesis_parts = [f"Ran {total} custom prompt(s): {compliant} compliant, {non_compliant} non-compliant, {unclear} unclear."]
        if issue_labels:
            synthesis_parts.append("Issues: " + "; ".join(issue_labels[:5]) + ("..." if len(issue_labels) > 5 else "."))

        return {
            "summary": f"Custom prompt-based analysis using {total} user-defined check(s).",
            "synthesis": " ".join(synthesis_parts),
            "overall_risk": overall_risk,
            "status_counts": {
                "compliant": compliant,
                "non_compliant": non_compliant,
                "unclear": unclear,
            },
            "findings": findings,
            "gaps": [],
            "notes": [f"Analysis driven by {total} custom prompt(s) provided by the user."],
        }

    # Default single-pass (no custom prompts)
    target_excerpt = truncate_text(target_text, 12000)
    context_block = format_retrieved_chunks_for_prompt(retrieved_chunks[:12])

    raw = _retryable_ollama_call(
        messages=[
            {"role": "system", "content": PROMPT_BASED_ANALYSIS_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"USER QUESTION:\n{user_query}\n\n"
                f"TARGET DOCUMENT:\n{target_excerpt}\n\n"
                f"RETRIEVED CONTEXT PASSAGES:\n{context_block}"
            )},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    parsed = _safe_json_loads(raw)
    if not parsed:
        return {
            "summary": "Prompt-based analysis could not be parsed.",
            "synthesis": raw.strip() if raw else "Prompt-based analysis did not return usable output.",
            "overall_risk": "unclear",
            "status_counts": {
                "compliant": 0,
                "non_compliant": 0,
                "unclear": 0,
            },
            "findings": [],
            "gaps": ["The model response could not be parsed as valid JSON."],
            "notes": [],
        }

    return _normalize_prompt_based_result(parsed)


def analyze_target_with_ollama(
    user_query: str,
    target_text: str,
    retrieved_chunks: List[Dict[str, Any]],
    analysis_mode: str = "compliance",
    execution_strategy: str = DEFAULT_EXECUTION_STRATEGY,
    custom_prompts: Optional[List[str]] = None,
    on_progress=None,
) -> Dict[str, Any]:
    """
    Multi-step agentic analysis using Ollama.

    Pipeline:
      1. Summarize the target document
      2. Evaluate each retrieved context chunk against the target
      3. Synthesize findings into a final report

    on_progress: optional callable(str) for streaming status updates.
    """
    def _progress(msg):
        if on_progress:
            on_progress(msg)

    execution_strategy = normalize_execution_strategy(execution_strategy)

    if execution_strategy == "prompt_based":
        _progress("Prompt-based analysis: assembling prompt...")
        result = _prompt_based_analyze_target(
            user_query=user_query,
            target_text=target_text,
            retrieved_chunks=retrieved_chunks,
            custom_prompts=custom_prompts or [],
            on_progress=on_progress,
        )
        _progress("Prompt-based analysis complete.")
        return result

    # Step 1: Summarize target
    _progress("Agent step 1/3: Summarizing target document...")
    target_summary = _agent_summarize_target(target_text)
    _progress(f"Target summary complete ({len(target_summary)} chars)")

    # Step 2: Evaluate each chunk
    findings = []
    target_excerpt = truncate_text(target_text, 3000)
    total = len(retrieved_chunks)

    for i, chunk in enumerate(retrieved_chunks, 1):
        _progress(f"Agent step 2/3: Evaluating context chunk {i}/{total} ({chunk.get('source_file', '')})...")
        finding = _agent_evaluate_chunk(
            user_query=user_query,
            target_summary=target_summary,
            target_excerpt=target_excerpt,
            chunk=chunk,
        )
        findings.append(finding)
        _progress(f"  Chunk {i} result: {finding['status']} — {finding['issue'][:80]}")

    # Step 3: Synthesize
    _progress("Agent step 3/3: Synthesizing final report...")
    synthesis = _agent_synthesize(user_query, target_summary, findings)

    # Count statuses
    compliant_count = sum(1 for f in findings if f["status"] == "compliant")
    non_compliant_count = sum(1 for f in findings if f["status"] == "non_compliant")
    unclear_count = sum(1 for f in findings if f["status"] == "unclear")

    if non_compliant_count > 0:
        overall_risk = "high"
    elif unclear_count > compliant_count:
        overall_risk = "medium"
    else:
        overall_risk = "low"

    return {
        "summary": target_summary,
        "synthesis": synthesis,
        "overall_risk": overall_risk,
        "status_counts": {
            "compliant": compliant_count,
            "non_compliant": non_compliant_count,
            "unclear": unclear_count,
        },
        "findings": [
            {
                "issue": f["issue"] or f"Chunk evaluation ({f['source_file']})",
                "status": f["status"],
                "risk_level": "high" if f["status"] == "non_compliant" else ("medium" if f["status"] == "unclear" else "low"),
                "explanation": f["explanation"],
                "context_evidence": [
                    {
                        "source_file": f["source_file"],
                        "chunk_id": f["chunk_id"],
                        "quote": f["evidence"],
                    }
                ],
            }
            for f in findings
            if f["issue"] or f["explanation"]
        ],
    }


# ---------------------------------------------------------------------
# Health / local runtime checks
# ---------------------------------------------------------------------

def check_ollama_health() -> Dict[str, Any]:
    """
    Check whether the active LLM provider appears reachable.

    With LLM_PROVIDER=claude (hosted deployments), there is no Ollama to probe —
    report on the Claude provider instead.
    """
    from ollama_executions import active_provider, claude_available, _get_claude_model

    if active_provider() == "claude":
        ok = claude_available()
        return {
            "ok": ok,
            "provider": "claude",
            "model": _get_claude_model(),
            "error": None if ok else "ANTHROPIC_API_KEY not set or anthropic package missing",
            "models": [_get_claude_model()] if ok else [],
        }

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip('"').strip()
    if not base_url:
        base_url = "http://localhost:11434"

    # Use the native Ollama endpoint for health/tags.
    tags_url = base_url.rstrip("/") + "/api/tags"
    if tags_url.endswith("/v1/api/tags"):
        tags_url = tags_url.replace("/v1/api/tags", "/api/tags")

    try:
        with urllib.request.urlopen(tags_url, timeout=3) as response:
            body = response.read().decode("utf-8", errors="ignore")
            parsed = json.loads(body) if body else {}
            models = [m.get("name", "") for m in parsed.get("models", [])]

            return {
                "ok": True,
                "base_url": base_url,
                "tags_url": tags_url,
                "models": models,
            }
    except Exception as exc:
        return {
            "ok": False,
            "base_url": base_url,
            "tags_url": tags_url,
            "error": str(exc),
            "models": [],
        }
    def _safe_int(value: Any, fallback: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback
