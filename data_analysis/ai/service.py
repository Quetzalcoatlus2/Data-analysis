# mypy: ignore-errors
"""Standalone AI summary / Q&A service.

Provides functions for generating AI summaries and answering user questions
about datasets using the Gemini API.  All functions accept explicit parameters
(logger, call_gemini_fn, etc.) instead of accessing module-level globals.
"""
from __future__ import annotations

import contextlib
import os
import re
from collections.abc import Callable
from typing import Any

import pandas as pd

from data_analysis.ai import engine as ai_engine
from data_analysis.ai import html_format as ai_html

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_describer(
    fn: Callable[[pd.DataFrame], str] | None,
) -> Callable[[pd.DataFrame], str]:
    if fn is not None:
        return fn
    try:
        from data_analysis.analysis.context import describe_for_ai
        return describe_for_ai
    except ImportError:
        return lambda _df: ''

def _get_offline(fn):
    if fn is not None:
        return fn
    try:
        from data_analysis.analysis.context import offline_answer
        return offline_answer
    except ImportError:
        return lambda df, q, error=None, filename=None: f'Offline: {error}'

# ---------------------------------------------------------------------------
# Q&A cache key builder
# ---------------------------------------------------------------------------

def _build_qna_cache_key(
    df: pd.DataFrame,
    question: str,
    filename: str | None = None,
) -> tuple[Any, ...]:
    """Build a cache key that is stable per question and dataset identity."""
    q_norm = (question or '').strip().lower()
    if filename:
        return ("file", str(filename), q_norm)

    if isinstance(df, pd.DataFrame):
        try:
            dataset_sig = (
                tuple(df.shape),
                tuple(str(c) for c in df.columns),
                tuple(str(t) for t in df.dtypes.to_list()),
                type(df.index).__name__,
            )
        except Exception:
            dataset_sig = (tuple(df.shape),)
    else:
        dataset_sig = (None, None)
    return ("df", dataset_sig, q_norm)


def _current_model_marker() -> str:
    """Build a stable attribution marker for the model used in this response."""
    model_name = (
        ai_engine.CURRENT_MODEL_NAME
        or ai_engine.AI_STATUS.get("model")
        or ai_engine.DEFAULT_AI_MODEL
        or "gemini-3-flash-preview"
    )
    return f"<!-- model:{model_name} -->"


def _attach_model_marker(html: str) -> str:
    """Ensure HTML contains one model marker so attribution survives sanitization and reloads."""
    text = str(html or "")
    if re.search(r"<!--\s*model:.*?-->", text):
        return text
    return f"{text}\n{_current_model_marker()}"


def _ensure_cached_summary_marker(html: str) -> str:
    """Backfill missing model marker on legacy cached summaries."""
    text = str(html or "")
    if not text.strip():
        return text
    if re.search(r"<!--\s*model:.*?-->", text):
        return text
    if ai_engine._is_offline_html(text):
        return text
    return _attach_model_marker(text)


# ---------------------------------------------------------------------------
# AI Summary
# ---------------------------------------------------------------------------

def get_ai_summary_with_file(
    df: pd.DataFrame,
    file_asset: Any = None,
    extra_context: str = "",
    *,
    describe_for_ai_fn: Callable[[pd.DataFrame], str] | None = None,
    sanitize_ai_html_fn: Callable[[str], str] | None = None,
    offline_answer_fn: Callable[..., str] | None = None,
    app_config: dict[str, Any] | None = None,
    logger: Any | None = None,
) -> str:
    sanitize = sanitize_ai_html_fn or ai_html.sanitize_ai_html

    if not ai_engine.ensure_ai_ready(logger=logger):
        return "AI analysis is disabled."

    describer = _get_describer(describe_for_ai_fn)
    try:
        df_desc = describer(df)
    except Exception:
        df_desc = ""

    prompt = (
        "You are a senior data analyst with expertise in statistical analysis and predictive insights. "
        "Produce a comprehensive, insightful HTML analysis of this dataset. "
        "Respond strictly in HTML (no Markdown) using <p>, <ul><li>, <table><thead><tbody><tr><th><td>, <strong>, <em>, <h3> for section headers. "
        "Structure your analysis into clearly separated sections with headers:\n\n"

        "<h3>📊 Dataset Overview</h3>\n"
        "Describe what this dataset captures, its size (rows × columns), time span if temporal, and primary purpose/domain.\n\n"

        "<h3>🔍 Data Quality Assessment</h3>\n"
        "Analyze: missing values (% per key columns), constant/near-constant columns, data types, suspected quality issues, "
        "duplicate records, outliers. Rate overall quality (Excellent/Good/Fair/Poor) with brief justification.\n\n"

        "<h3>📈 Key Statistical Insights</h3>\n"
        "Provide distributions (mean, median, std dev), ranges (min/max), skewness. Identify interesting patterns, "
        "dominant segments, notable ratios. Use concrete numbers. Highlight any surprising findings.\n\n"

        "<h3>⏱️ Temporal Patterns & Trends</h3>\n"
        "If time-indexed: identify trends (upward/downward/stable), seasonality (daily/weekly/monthly cycles), "
        "volatility periods, growth rates. If not temporal, state '<p>Not applicable (no datetime index).</p>'\n\n"

        "<h3>🔗 Correlations & Relationships</h3>\n"
        "Highlight strongest positive and negative correlations (cite correlation coefficients). "
        "Explain what these relationships might indicate. Identify potential causal links vs. coincidences.\n\n"

        "<h3>⚠️ Anomalies & Irregularities</h3>\n"
        "Flag unusual spikes, dips, gaps, or inconsistencies. Estimate their severity and potential causes. "
        "If none found, state clearly.\n\n"

        "<h3>🔮 Prognosis & Future Outlook</h3>\n"
        "Based on observed trends and patterns, what are plausible future directions? Identify: "
        "expected continuations, potential risks, growth/decline scenarios. Be specific but qualify uncertainty. "
        "State confidence level (high/medium/low) for each prediction.\n\n"

        "<h3>💡 Actionable Observations</h3>\n"
        "Provide 3-5 concrete, prioritized recommendations for: data collection improvements, further analyses, "
        "decision-making insights, or intervention points. Use bullet points.\n\n"

        "<h3>⚠️ Limitations & Caveats</h3>\n"
        "Acknowledge: sampling biases, data sparsity, interpretability constraints, assumptions made, "
        "or factors that could invalidate conclusions.\n\n"

        "Guidelines:\n"
        "- Be analytical and precise—cite concrete numbers (counts, percentages, ranges, correlation coefficients)\n"
        "- Focus on conclusions and insights, not just descriptions\n"
        "- Explain the 'so what'—why findings matter\n"
        "- Be honest about uncertainty\n"
        "- Avoid jargon; explain technical terms briefly\n"
        "- Do NOT hallucinate fields or values not in the context\n"
        "- Keep professional tone; avoid marketing language\n\n"

        "Context:\n" + df_desc + ("\n\nAdditional context:\n" + extra_context if extra_context else "")
    )

    gen_cfg = {
        "max_output_tokens": 16384,
        "temperature": 0.4,
        "top_p": 0.95,
        "top_k": 40,
        "response_mime_type": "text/plain",
    }

    cfg = app_config or {}

    try:
        resp = ai_engine._call_gemini(prompt, file_asset=file_asset, generation_config=gen_cfg, app_config=cfg, logger=logger)
        text = ai_engine._extract_text_from_gemini_response(resp, logger=logger).strip()
        if not text:
            try:
                diag = ai_engine._diagnose_gemini_response(resp)
                if diag and logger:
                    logger.warning("AI summary empty response: %s", diag)
            except Exception as diag_err:
                if logger:
                    logger.debug("AI summary diagnostics unavailable: %s", diag_err)
            try:
                simple_cfg = {"max_output_tokens": 384, "temperature": 0.2, "response_mime_type": "text/plain"}
                simple_prompt = "Provide a concise HTML summary of the dataset using <p> and <ul><li> only."
                simple = ai_engine._call_gemini(simple_prompt + "\n\n" + extra_context, file_asset=file_asset, generation_config=simple_cfg, app_config=cfg, logger=logger)
                text2 = ai_engine._extract_text_from_gemini_response(simple, logger=logger).strip()
                if text2:
                    return _attach_model_marker(sanitize(text2))
            except Exception as simple_err:
                if logger:
                    logger.debug("AI summary simple fallback failed: %s", simple_err)
            try:
                resp2 = ai_engine._call_gemini(prompt, file_asset=None, generation_config={**gen_cfg, "max_output_tokens": 768}, app_config=cfg, logger=logger)
                text_only = ai_engine._extract_text_from_gemini_response(resp2, logger=logger).strip()
                if text_only:
                    return _attach_model_marker(sanitize(text_only))
            except Exception as e2:
                if logger:
                    logger.info("Text-only fallback failed for AI summary: %s", e2)
            d = ai_engine._diagnose_gemini_response(resp)
            raise RuntimeError("Empty AI response" + (f" ({d})" if d else ""))

        try:
            fr = ai_engine._get_finish_reason(resp)
            if isinstance(fr, str) and "MAX_TOKENS" in fr:
                cont_prompt = (
                    "Continue the same HTML summary in the same style. Do not repeat previous text. "
                    "Only output valid HTML fragments (<p>, <ul><li>, <table>).\n\n"
                    f"Previous tail for context (do not repeat):\n{text[-1200:]}"
                )
                cont = ai_engine._call_gemini(cont_prompt, file_asset=file_asset, generation_config={
                    "max_output_tokens": 1024, "temperature": 0.3, "top_p": 0.95,
                    "top_k": 40, "response_mime_type": "text/plain",
                }, app_config=cfg, logger=logger)
                more = ai_engine._extract_text_from_gemini_response(cont, logger=logger).strip()
                if more:
                    text = text + "\n" + more
        except Exception as ce:
            if logger:
                logger.info("Summary continuation skipped: %s", ce)

        with contextlib.suppress(Exception):
            text = _attach_model_marker(text)

        sanitized = sanitize(text)

        # Re-append model marker after sanitize (sanitize strips HTML comments)
        try:
            if '<!-- model:' not in sanitized:
                sanitized = _attach_model_marker(sanitized)
        except Exception:
            pass

        return sanitized
    except Exception as e:
        if logger:
            logger.warning("AI summary failed, falling back. Error: %s", e)
        if offline_answer_fn:
            return _get_offline(offline_answer_fn)(df, "summary", error=e)
        return f"<p>AI summary unavailable: {e}</p>"


def get_ai_answer_with_file(
    df: pd.DataFrame,
    question: str,
    file_asset: Any = None,
    filename: str | None = None,
    *,
    describe_for_ai_fn: Callable[[pd.DataFrame], str] | None = None,
    sanitize_ai_html_fn: Callable[[str], str] | None = None,
    offline_answer_fn: Callable[..., str] | None = None,
    get_clean_ai_summary_fn: Callable[[str], str | None] | None = None,
    qna_cache: Any | None = None,
    app_config: dict[str, Any] | None = None,
    logger: Any | None = None,
) -> str:
    """Answer a user's question about the dataset."""
    sanitize = sanitize_ai_html_fn or ai_html.sanitize_ai_html
    cfg = app_config or {}

    try:
        cache_key = None
        try:
            cache_key = _build_qna_cache_key(df, question, filename=filename)
            if qna_cache is not None:
                cached = qna_cache.get(cache_key)
                if isinstance(cached, str) and cached.strip():
                    return cached
        except Exception as cache_read_err:
            if logger:
                logger.debug("Q&A cache read skipped due to key/cache error: %s", cache_read_err)

        if not ai_engine.ensure_ai_ready(logger=logger):
            if offline_answer_fn:
                return _get_offline(offline_answer_fn)(df, question, error="AI disabled.", filename=filename)
            return "<p>AI is disabled.</p>"

        summary_html = None
        if filename and get_clean_ai_summary_fn:
            try:
                summary_html = get_clean_ai_summary_fn(filename)
            except Exception as summary_cache_err:
                if logger:
                    logger.debug("Q&A summary cache lookup failed for %s: %s", filename, summary_cache_err)
                summary_html = None

        describer = _get_describer(describe_for_ai_fn)

        if summary_html:
            try:
                context_text = re.sub(r"<[^>]+>", " ", str(summary_html))
                context_text = re.sub(r"\s+", " ", context_text).strip()
            except Exception:
                context_text = describer(df)
        else:
            context_text = describer(df)

        prompt = f"""
You are an expert data analyst. Your job is to answer questions about this dataset with CONFIDENCE.

ABSOLUTE RULES:
1. ALWAYS provide your best answer. NEVER refuse to answer or say "I cannot determine this."
2. If you're uncertain, give your best interpretation and add a brief disclaimer at the end like: "Note: This interpretation is based on available data patterns." (Do NOT say "based on the first 10 rows" - you are provided with statistics covering the ENTIRE dataset).
3. Use SPECIFIC numbers, percentages, and country names from the data as evidence.
4. State findings as FACTS when the data supports them. Avoid weak language like "might", "could be", "possibly".
5. For "why" questions: provide the most likely explanation based on data patterns, then note it's an interpretation.

RESPONSE STYLE:
- Lead with the direct answer
- Support with specific data points
- Add brief disclaimer ONLY at the end if truly needed
- Never refuse, always give your best analysis

FORMAT: HTML only (<p>, <ul><li>, <table>, <strong>, <em>, <h4>). No markdown.

Dataset Context:
{context_text}

Question:
{question}
""".strip()

        resp = ai_engine._call_gemini(prompt, file_asset=file_asset, generation_config={
            "max_output_tokens": 8192, "temperature": 0.4, "top_p": 0.95,
            "top_k": 40, "response_mime_type": "text/plain",
        }, app_config=cfg, logger=logger)
        text = ai_engine._extract_text_from_gemini_response(resp, logger=logger).strip()

        if not text:
            try:
                diag = ai_engine._diagnose_gemini_response(resp)
                if diag and logger:
                    logger.warning("AI Q&A empty response: %s", diag)
            except Exception as diag_err:
                if logger:
                    logger.debug("AI Q&A diagnostics unavailable: %s", diag_err)
            try:
                simple = ai_engine._call_gemini(
                    f"Answer briefly in HTML (<p>, <ul><li>) only: {question}\n\nContext:\n{context_text}",
                    file_asset=file_asset,
                    generation_config={"max_output_tokens": 384, "temperature": 0.2, "response_mime_type": "text/plain"},
                    app_config=cfg, logger=logger,
                )
                text2 = ai_engine._extract_text_from_gemini_response(simple, logger=logger).strip()
                if text2:
                    html2 = _attach_model_marker(sanitize(text2))
                    try:
                        if cache_key and qna_cache is not None:
                            qna_cache.set(cache_key, html2)
                    except Exception:
                        pass
                    return html2
            except Exception as simple_err:
                if logger:
                    logger.debug("AI Q&A simple fallback failed: %s", simple_err)
            try:
                resp2 = ai_engine._call_gemini(prompt, file_asset=None, generation_config={
                    "max_output_tokens": 640, "temperature": 0.25, "top_p": 0.95,
                    "top_k": 40, "response_mime_type": "text/plain",
                }, app_config=cfg, logger=logger)
                text3 = ai_engine._extract_text_from_gemini_response(resp2, logger=logger).strip()
                if text3:
                    html3 = _attach_model_marker(sanitize(text3))
                    try:
                        if cache_key and qna_cache is not None:
                            qna_cache.set(cache_key, html3)
                    except Exception:
                        pass
                    return html3
            except Exception as e2:
                if logger:
                    logger.info("Text-only fallback failed for AI Q&A: %s", e2)
            d = ai_engine._diagnose_gemini_response(resp)
            if offline_answer_fn:
                return _get_offline(offline_answer_fn)(df, question, error=("Empty AI response" + (f" ({d})" if d else "")), filename=filename)
            return "<p>AI response unavailable.</p>"

        try:
            fr = ai_engine._get_finish_reason(resp)
            if isinstance(fr, str) and "MAX_TOKENS" in fr:
                if logger:
                    logger.info("Q&A response truncated (MAX_TOKENS), attempting continuation...")
                cont_prompt = (
                    "Continue the same HTML answer to the user's question. Do not repeat previous text. "
                    "Only output valid HTML fragments (<p>, <ul><li>, <table>).\n\n"
                    f"Question: {question}\n"
                    f"Previous tail for context (do not repeat):\n{text[-900:]}"
                )
                try:
                    cont = ai_engine._call_gemini(cont_prompt, file_asset=file_asset, generation_config={
                        "max_output_tokens": 1536, "temperature": 0.25, "top_p": 0.95,
                        "top_k": 40, "response_mime_type": "text/plain",
                    }, app_config=cfg, logger=logger)
                    more = ai_engine._extract_text_from_gemini_response(cont, logger=logger).strip()
                    if more:
                        text = text + "\n" + more
                        if logger:
                            logger.info("Q&A continuation successful")
                    else:
                        if logger:
                            logger.warning("Q&A continuation returned empty, using truncated response")
                        text = text + "\n<p><em>(Response was truncated due to length)</em></p>"
                except Exception as cont_err:
                    if logger:
                        logger.warning("Q&A continuation failed: %s, using truncated response", cont_err)
                    text = text + "\n<p><em>(Response was truncated due to length)</em></p>"
        except Exception as ce:
            if logger:
                logger.info("Q&A continuation check skipped: %s", ce)

        html = _attach_model_marker(sanitize(text))

        try:
            if cache_key and qna_cache is not None and not ai_engine._is_offline_html(html):
                qna_cache.set(cache_key, html)
        except Exception:
            pass
        return html

    except Exception as e:
        if logger:
            logger.warning("AI Q&A failed; falling back. Error: %s", e)
        if offline_answer_fn:
            return _get_offline(offline_answer_fn)(df, question, error=e, filename=filename)
        return f"<p>AI answer unavailable: {e}</p>"


def get_or_cache_ai_summary_for(
    filename: str,
    df: pd.DataFrame,
    extra_context: str = "",
    *,
    ai_summary_cache: dict[str, Any] | None = None,
    ai_file_map: dict[str, Any] | None = None,
    describe_for_ai_fn: Callable[..., str] | None = None,
    sanitize_ai_html_fn: Callable[[str], str] | None = None,
    offline_answer_fn: Callable[..., str] | None = None,
    app_config: dict[str, Any] | None = None,
    logger: Any | None = None,
) -> str:
    """Return the AI summary for this file, generating it once and caching it."""
    try:
        # Check memory cache first
        if ai_summary_cache is not None:
            cached = ai_summary_cache.get(filename)
            if isinstance(cached, str) and cached.strip():
                with_marker = _ensure_cached_summary_marker(cached)
                if with_marker != cached:
                    ai_summary_cache[filename] = with_marker
                return with_marker
                
        # Check disk cache
        disk_cached = None
        path = None
        if app_config:
            uploads_dir = app_config.get('UPLOADS_DIR', 'datasets')
            safe_name = filename.replace('/', '_').replace('\\', '_')
            path = os.path.join(uploads_dir, f"{safe_name}.ai_summary.html")
            if os.path.exists(path):
                try:
                    with open(path, encoding='utf-8') as f:
                        disk_cached = f.read()
                except Exception:
                    pass
        
        if disk_cached and isinstance(disk_cached, str) and disk_cached.strip():
            disk_cached = _ensure_cached_summary_marker(disk_cached)
            if ai_summary_cache is not None:
                ai_summary_cache[filename] = disk_cached
            if path:
                try:
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(disk_cached)
                except Exception:
                    pass
            return disk_cached

        file_asset = (ai_file_map or {}).get(filename)
        ai_html_result = get_ai_summary_with_file(
            df, file_asset=file_asset, extra_context=extra_context,
            describe_for_ai_fn=describe_for_ai_fn,
            sanitize_ai_html_fn=sanitize_ai_html_fn,
            offline_answer_fn=offline_answer_fn,
            app_config=app_config, logger=logger,
        )
        if isinstance(ai_html_result, str) and not ai_engine._is_offline_html(ai_html_result):
            if ai_summary_cache is not None:
                ai_summary_cache[filename] = ai_html_result
            if path:
                try:
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(ai_html_result)
                except Exception as e:
                    if logger:
                        logger.warning("Failed to write AI summary to disk: %s", e)
        return ai_html_result
    except Exception as e:
        return f"<p>AI summary unavailable: {e}</p>"


def _get_clean_ai_summary_from_cache(
    filename: str,
    *,
    ai_summary_cache: dict[str, Any] | None = None,
    sanitize_ai_html_fn: Callable[[str], str] | None = None,
    app_config: dict[str, Any] | None = None,
) -> str | None:
    """Read AI summary from cache and sanitize legacy/broken fragments if present."""
    sanitize = sanitize_ai_html_fn or ai_html.sanitize_ai_html
    try:
        cached = None
        if ai_summary_cache is not None:
            cached = ai_summary_cache.get(filename)
            
        path = None
        if not cached and app_config:
            uploads_dir = app_config.get('UPLOADS_DIR', 'datasets')
            safe_name = filename.replace('/', '_').replace('\\', '_')
            path = os.path.join(uploads_dir, f"{safe_name}.ai_summary.html")
            if os.path.exists(path):
                try:
                    with open(path, encoding='utf-8') as f:
                        cached = f.read()
                except Exception:
                    pass

        if not isinstance(cached, str) or not cached.strip():
            return None
            
        # Extract model marker before sanitization
        model_marker = None
        _m = re.search(r'<!--\s*model:(.*?)\s*-->', cached)
        if _m:
            model_marker = _m.group(0)
            
        cleaned = sanitize(cached)

        # Backfill marker for legacy cache entries that predate marker embedding.
        cleaned = _ensure_cached_summary_marker(cleaned)

        # Re-append model marker if it was stripped
        if model_marker and '<!-- model:' not in cleaned:
            cleaned = cleaned + f"\n{model_marker}"
            
        if cleaned != cached:
            if ai_summary_cache is not None:
                ai_summary_cache[filename] = cleaned
            if path:
                try:
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(cleaned)
                except Exception:
                    pass
        return cleaned
    except Exception:
        if ai_summary_cache is not None:
            val = ai_summary_cache.get(filename)
            return val if isinstance(val, str) else None
        return None


__all__ = [
    "_build_qna_cache_key",
    "get_ai_summary_with_file",
    "get_ai_answer_with_file",
    "get_or_cache_ai_summary_for",
    "_get_clean_ai_summary_from_cache",
]
