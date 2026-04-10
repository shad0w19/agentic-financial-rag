# -*- coding: utf-8 -*-
import streamlit as st
import sys
import os
import re
import uuid
import logging
import json
import subprocess
from datetime import datetime
from pathlib import Path
import pandas as pd
from src.config.settings import GENERAL_MODEL, REASONING_MODEL

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logger = logging.getLogger(__name__)
APP_ROOT = Path(__file__).resolve().parent
REPORT_FAST_PATH = APP_ROOT / "reports" / "smoke_latency_fast_only.json"
REPORT_DEEP_PATH = APP_ROOT / "reports" / "deep_sample_results.json"
RAW_DOMAIN_TARGETS = {
    "Personal Tax": APP_ROOT / "data" / "raw" / "income_tax",
    "Corporate Tax": APP_ROOT / "data" / "raw" / "corporate_tax",
    "GST": APP_ROOT / "data" / "raw" / "gst",
    "Investment": APP_ROOT / "data" / "raw" / "investment",
    "Regulatory": APP_ROOT / "data" / "raw" / "regulatory",
}
KB_DOMAIN_CONFIG = [
    (
        "Personal Tax",
        APP_ROOT / "data" / "vector_store" / "personal_tax" / "metadata.json",
        "Income tax, deductions, ITR, HRA",
    ),
    (
        "Corporate Tax",
        APP_ROOT / "data" / "vector_store" / "corporate_tax" / "metadata.json",
        "Company income, MAT, slabs",
    ),
    (
        "GST",
        APP_ROOT / "data" / "vector_store" / "gst" / "metadata.json",
        "Rates, registration, ITC",
    ),
    (
        "Investment",
        APP_ROOT / "data" / "vector_store" / "investment" / "metadata.json",
        "Mutual funds, SIP, stocks, retirement",
    ),
    (
        "Regulatory",
        APP_ROOT / "data" / "vector_store" / "regulatory" / "metadata.json",
        "SEBI, FEMA, DTAA, compliance circulars",
    ),
]

st.set_page_config(
    page_title="FinAdvisor | AI Workspace",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── IN-MEMORY SESSION MANAGEMENT ─────────────────────────────────────────────
if "sessions" not in st.session_state:
    default_id = str(uuid.uuid4())
    st.session_state.sessions = {
        default_id: {
            "id": default_id,
            "title": "New Conversation",
            "messages": [],
            "created_at": datetime.now().strftime("%I:%M %p")
        }
    }
    st.session_state.current_session_id = default_id

if "active_tool" not in st.session_state:
    st.session_state.active_tool = None

if "ingestion_status" not in st.session_state:
    st.session_state.ingestion_status = None

if "last_staged_upload" not in st.session_state:
    st.session_state.last_staged_upload = None

def create_new_session():
    new_id = str(uuid.uuid4())
    st.session_state.sessions[new_id] = {
        "id": new_id,
        "title": f"New Conversation {len(st.session_state.sessions) + 1}",
        "messages": [],
        "created_at": datetime.now().strftime("%I:%M %p")
    }
    st.session_state.current_session_id = new_id

def switch_session(session_id):
    st.session_state.current_session_id = session_id

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif;
    background: #F8FAFC;
    color: #0F172A;
}

#MainMenu, footer, header { visibility: hidden; }

/* ─── SLIM HEADER ─── */
.custom-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 12px 0; margin-bottom: 16px;
    border-bottom: 1px solid #E2E8F0;
    position: sticky; top: 0; z-index: 100;
    background: rgba(248, 250, 252, 0.95);
    backdrop-filter: blur(8px);
}
.header-left { display: flex; align-items: center; gap: 12px; }
.logo-box {
    width: 28px; height: 28px; background: #2563EB;
    border-radius: 6px; display: flex; align-items: center; justify-content: center;
    box-shadow: 0 2px 4px rgba(37, 99, 235, 0.15);
}
.logo-box svg { width: 14px; height: 14px; fill: white; }
.header-title { font-size: 16px; font-weight: 700; color: #0F172A; margin: 0; letter-spacing: -0.02em; }
.header-subtitle { font-size: 10px; color: #94A3B8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; margin: 0; }
.status-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: #F0FDF4; color: #166534; font-size: 10px; font-weight: 700;
    padding: 4px 10px; border-radius: 9999px; border: 1px solid #DCFCE7;
}
.status-dot { width: 6px; height: 6px; background: #22C55E; border-radius: 50%; }

/* ─── TABS ─── */
div[data-testid="stTabs"] [data-testid="stTabBar"] {
    background: transparent; border-bottom: 1px solid #E2E8F0; gap: 0; padding: 0;
}
div[data-testid="stTabs"] button[role="tab"] {
    font-family: 'Inter', sans-serif !important; font-size: 13px !important; font-weight: 500 !important;
    color: #64748B !important; padding: 10px 20px !important;
    border-bottom: 2px solid transparent !important; background: transparent !important;
}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #0F172A !important; border-bottom: 2px solid #2563EB !important; font-weight: 600 !important;
}

/* ─── COMPONENT STYLING ─── */
/* Selectbox container */
.stSelectbox > div > div {
    background: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
}
/* Selected value text — white bg, dark text */
.stSelectbox [data-baseweb="select"] [data-baseweb="value"] *,
.stSelectbox [data-baseweb="select"] > div > div {
    color: #0F172A !important;
    background: #FFFFFF !important;
    font-weight: 500 !important;
}
/* Fix white-on-white dropdown text */
.stSelectbox div[data-baseweb="select"] span,
.stSelectbox div[data-baseweb="select"] div,
[data-baseweb="select"] [data-baseweb="value"] {
    color: #0F172A !important;
    font-weight: 500 !important;
}
[data-baseweb="popover"] li {
    color: #0F172A !important;
    background: #FFFFFF !important;
}
[data-baseweb="popover"] li:hover {
    background: #F1F5F9 !important;
}
.stFileUploader { background: #F8FAFC; border: 1px dashed #CBD5E1; border-radius: 8px; padding: 16px; }

/* Tighter spacing for the engine description */
.engine-descriptor { font-size: 12px; color: #64748B; margin-top: -12px; margin-bottom: 24px; padding-left: 4px; line-height: 1.4; }
.sidebar-title { font-size: 11px; font-weight: 600; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 24px; margin-bottom: 12px; }

/* Professional Stacked Sidebar Buttons */
[data-testid="stSidebar"] .stButton > button {
    background-color: #EFF6FF !important;
    border: 1px solid #BFDBFE !important;
    color: #1D4ED8 !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 10px 16px !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.01em !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    border-color: #2563EB !important;
    background-color: #DBEAFE !important;
    color: #1E40AF !important;
}
/* ─── KILL TOP PADDING / SCROLL ─── */
.block-container {
    padding-top: 0.5rem !important;
    padding-bottom: 0 !important;
    max-width: 100% !important;
}
[data-testid="stAppViewContainer"] {
    overflow: hidden !important;
}
section[data-testid="stSidebar"] {
    top: 0 !important;
}
div[data-testid="stToolbar"] {
    display: none !important;
}
/* ─── CHAT THREAD ─── */
.msg-user-row { display: flex; justify-content: flex-end; margin-bottom: 24px; }
.msg-user-bubble {
    background: #F1F5F9; color: #0F172A; border-radius: 12px;
    padding: 12px 18px; max-width: 80%; font-size: 14.5px; line-height: 1.5;
}
.msg-ai-row { display: flex; gap: 16px; margin-bottom: 32px; align-items: flex-start; }
.msg-ai-avatar {
    width: 32px; height: 32px; background: #2563EB; border-radius: 8px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center; margin-top: 2px;
}
.msg-ai-avatar svg { width: 16px; height: 16px; fill: white; }
.msg-ai-body { flex: 1; min-width: 0; }
.msg-ai-text { font-size: 14.5px; line-height: 1.6; color: #1E293B; background: transparent; padding: 0; }

/* ─── PILL TELEMETRY ─── */
.telemetry-row { margin-top: 12px; display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.telemetry-pill {
    background: #FFFFFF; border: 1px solid #E2E8F0; color: #64748B;
    font-size: 11px; font-weight: 500; padding: 4px 10px; border-radius: 999px; display: flex; align-items: center; gap: 4px;
}
.telemetry-pill.conf-high { border-color: #BBF7D0; background: #F0FDF4; color: #166534; }
.telemetry-pill span { font-family: 'Courier New', monospace; font-weight: 600; color: #0F172A; }

/* ─── CARDS & ARCHITECTURE ─── */
.sec-col, .analytic-card { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 20px; height: 100%; box-shadow: 0 1px 3px rgba(0,0,0,0.02); }
.sec-col-title { font-size: 14px; font-weight: 600; color: #0F172A; margin: 0 0 6px 0; }
.sec-col-desc { font-size: 12px; color: #64748B; margin: 0 0 14px 0; line-height: 1.5; }
.threat-item { font-size: 12px; color: #475569; padding: 5px 0; border-top: 1px solid #F1F5F9; line-height: 1.4; }
.threat-label { display: inline-block; background: #FEF2F2; color: #991B1B; font-size: 10px; font-weight: 600; padding: 1px 6px; border-radius: 3px; margin-right: 6px; font-family: 'Courier New', monospace; text-transform: uppercase; }

/* ─── DETAILED PIPELINE CSS ─── */
.pipeline-wrap { display: flex; align-items: center; justify-content: center; gap: 0; padding: 24px 0; overflow-x: auto; }
.pipe-step { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px 18px; text-align: center; min-width: 130px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
.pipe-step-num { font-family: 'Courier New', monospace; font-size: 10px; font-weight: 700; color: #2563EB; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }
.pipe-step-name { font-size: 13px; font-weight: 600; color: #0F172A; margin-bottom: 4px; }
.pipe-step-desc { font-size: 11px; color: #94A3B8; line-height: 1.4; }
.pipe-arrow { color: #CBD5E1; font-size: 18px; padding: 0 4px; flex-shrink: 0; }

.domain-card { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px 20px; margin-bottom: 10px; }
.domain-name { font-size: 13px; font-weight: 600; color: #0F172A; margin-bottom: 4px; }
.domain-chunks { font-family: 'Courier New', monospace; font-size: 22px; font-weight: 700; color: #2563EB; }
.domain-label { font-size: 11px; color: #94A3B8; margin-left: 4px; }
.tech-row { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #F1F5F9; }

/* ─── ANALYTICS SPECIFIC ─── */
.analytic-value { font-family: 'Courier New', monospace; font-size: 28px; font-weight: 700; color: #2563EB; margin: 8px 0; }
.analytic-label { font-size: 12px; font-weight: 600; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em; }
.analytic-trend-up { color: #16A34A; font-size: 12px; font-weight: 500; display: flex; align-items: center; gap: 4px;}
.analytic-trend-down { color: #DC2626; font-size: 12px; font-weight: 500; display: flex; align-items: center; gap: 4px;}

</style>
""", unsafe_allow_html=True)

# ─── BACKEND DEMO LOADER ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_demo():
    try:
        from src.services.query_orchestrator import QueryOrchestrator
        orchestrator = QueryOrchestrator(preload_faiss=True)
        try:
            orchestrator._ensure_workflow()
            orchestrator.run_query("warmup", timeout_seconds=10)
        except Exception:
            pass
        return orchestrator, None
    except Exception as e:
        return None, str(e)

demo, _ = load_demo()

def clean_answer(raw):
    if not raw: return "No answer generated."
    final_match = re.search(r"final\s*answer\s*:\s*", raw, flags=re.IGNORECASE)
    if final_match: return raw[final_match.end():].strip()
    return raw.strip()

@st.cache_resource(show_spinner=False)
def load_tax_service():
    from src.import_map import DeductionBreakdown, IncomeBreakdown
    from src.services.tax_calculator_service import TaxCalculatorService

    return TaxCalculatorService(), IncomeBreakdown, DeductionBreakdown

@st.cache_resource(show_spinner=False)
def load_investment_service():
    from src.services.investment_service import InvestmentService

    return InvestmentService()

@st.cache_data(show_spinner=False)
def load_json_file(path_str):
    path = Path(path_str)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to load JSON file: %s", path)
        return {}

@st.cache_data(show_spinner=False)
def load_knowledge_base_stats():
    stats = []
    for label, metadata_path, description in KB_DOMAIN_CONFIG:
        metadata = load_json_file(str(metadata_path))
        if isinstance(metadata, dict):
            chunk_count = len(metadata)
        elif isinstance(metadata, list):
            chunk_count = len(metadata)
        else:
            chunk_count = 0
        stats.append(
            {
                "name": label,
                "chunks": chunk_count,
                "description": description,
            }
        )
    return stats

def format_currency(value):
    return f"Rs. {value:,.0f}"

def format_latency_ms(value):
    if value is None:
        return "n/a"
    seconds = value / 1000
    return f"{seconds:.1f}s"

def sanitize_filename(filename):
    return re.sub(r"[^A-Za-z0-9._-]", "_", filename)

def stage_uploaded_document(uploaded_file, domain_label):
    target_dir = RAW_DOMAIN_TARGETS[domain_label]
    target_dir.mkdir(parents=True, exist_ok=True)
    stamped_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{sanitize_filename(uploaded_file.name)}"
    target_path = target_dir / stamped_name
    target_path.write_bytes(uploaded_file.getbuffer())
    return target_path

def rebuild_indices():
    command = [sys.executable, "-m", "src.data_pipeline.run_pipeline"]
    return subprocess.run(
        command,
        cwd=str(APP_ROOT),
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )

def build_analytics_snapshot():
    fast_report = load_json_file(str(REPORT_FAST_PATH))
    deep_report = load_json_file(str(REPORT_DEEP_PATH))

    fast_records = fast_report.get("records", []) if isinstance(fast_report, dict) else []
    deep_records = deep_report.get("records", []) if isinstance(deep_report, dict) else []
    fast_summary = fast_report.get("summary", {}) if isinstance(fast_report, dict) else {}
    fast_mode_summary = fast_summary.get("by_mode", {}).get("fast", {}) if isinstance(fast_summary, dict) else {}

    deep_latencies = [
        record.get("timings", {}).get("total")
        for record in deep_records
        if record.get("timings", {}).get("total") is not None
    ]

    all_records = fast_records + deep_records
    blocked_count = sum(1 for record in all_records if record.get("blocked"))
    timeout_count = sum(1 for record in all_records if record.get("timeout_stage"))
    degraded_count = sum(1 for record in all_records if record.get("degraded_flags"))

    mode_utilization = pd.DataFrame(
        {
            "queries": [len(fast_records), len(deep_records)],
        },
        index=["Fast Lane", "Deep Workflow"],
    )

    issue_distribution = pd.DataFrame(
        {
            "count": [blocked_count, timeout_count, degraded_count],
        },
        index=["Blocked", "Timeouts", "Degraded"],
    )

    return {
        "total_queries": len(all_records),
        "fast_avg_latency_ms": fast_mode_summary.get("latency_ms_mean"),
        "deep_avg_latency_ms": sum(deep_latencies) / len(deep_latencies) if deep_latencies else None,
        "block_rate": (blocked_count / len(all_records)) if all_records else 0.0,
        "lane_match_rate": fast_summary.get("lane_match_rate", 0.0),
        "grounded_rate": fast_summary.get("grounded_retrieval_rate", 0.0),
        "table_rate": fast_summary.get("table_like_hit_rate", 0.0),
        "mode_utilization": mode_utilization,
        "issue_distribution": issue_distribution,
        "updated_at": datetime.now().strftime("%I:%M %p"),
    }

def render_ai_message(msg):
    answer_html = msg["content"].replace("\n\n", "</p><p>").replace("\n", "<br>")
    meta = msg.get("meta", {})
    conf = meta.get("confidence", 0)
    retrieved = meta.get("retrieved_docs_count", 0)
    total_s = meta.get("timings", {}).get("total", 0) / 1000
    plan_steps = meta.get("plan_steps", [])
    
    conf_class = "conf-high" if conf > 0.8 else ""
    
    st.markdown(f"""
    <div class="msg-ai-row">
      <div class="msg-ai-avatar"><svg viewBox="0 0 24 24"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg></div>
      <div class="msg-ai-body">
        <div class="msg-ai-text"><p>{answer_html}</p></div>
        <div class="telemetry-row">
            <div class="telemetry-pill {conf_class}">Confidence: <span>{conf:.0%}</span></div>
            <div class="telemetry-pill">Sources: <span>{retrieved}</span></div>
            <div class="telemetry-pill">Latency: <span>{total_s:.1f}s</span></div>
            <div class="telemetry-pill">Hops: <span>{len(plan_steps)}</span></div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)


# ─── SIDEBAR: CHAT MEMORY ─────────────────────────────────────────────────────
# ─── SIDEBAR: COMMAND CENTER & MEMORY ─────────────────────────────────────────
with st.sidebar:
    st.markdown("### FinAdvisor")
    
    # Session Management
    st.markdown('<p class="sidebar-title">Recent Chats</p>', unsafe_allow_html=True)
    if st.button("New Conversation", use_container_width=True, type="primary"):
        create_new_session()
        st.rerun()
        
    st.markdown("<br>", unsafe_allow_html=True)
    for s_id, session_data in reversed(st.session_state.sessions.items()):
        prefix = "— " if s_id == st.session_state.current_session_id else ""
        if st.button(f"{prefix}{session_data['title']}", key=f"btn_{s_id}", use_container_width=True):
            switch_session(s_id)
            st.rerun()

    st.divider()
    
    # Tools & Ingestion Section
    st.markdown('<p class="sidebar-title">Tools and Ingestion</p>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"], label_visibility="collapsed")
    if uploaded_file:
        st.caption(f"Ready to stage: {uploaded_file.name}")
        ingest_domain = st.selectbox(
            "Index into",
            options=list(RAW_DOMAIN_TARGETS.keys()),
            key="ingest_domain",
        )
        stage_col, index_col = st.columns(2)
        with stage_col:
            if st.button("Stage Upload", use_container_width=True):
                staged_path = stage_uploaded_document(uploaded_file, ingest_domain)
                st.session_state.last_staged_upload = str(staged_path)
                st.session_state.ingestion_status = f"Staged {staged_path.name} for {ingest_domain}."
                st.success("Document staged.", icon="✓")
        with index_col:
            if st.button("Rebuild Index", use_container_width=True):
                with st.spinner("Rebuilding vector indices..."):
                    result = rebuild_indices()
                if result.returncode == 0:
                    load_json_file.clear()
                    load_knowledge_base_stats.clear()
                    load_demo.clear()
                    st.session_state.ingestion_status = "Index rebuild completed successfully."
                    st.success("Indices rebuilt.", icon="✓")
                else:
                    error_tail = (result.stderr or result.stdout or "Unknown pipeline failure")[-400:]
                    st.session_state.ingestion_status = f"Index rebuild failed: {error_tail}"
                    st.error("Index rebuild failed.")
    elif st.session_state.ingestion_status:
        st.caption(st.session_state.ingestion_status)

    if st.session_state.last_staged_upload:
        staged_name = Path(st.session_state.last_staged_upload).name
        st.caption(f"Latest staged file: {staged_name}")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Stacked, professional tool buttons
    if st.button("Tax Calculator", use_container_width=True):
        st.session_state.active_tool = "tax"
        st.rerun()
    if st.button("Investment Planner", use_container_width=True):
        st.session_state.active_tool = "investment"
        st.rerun()

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="custom-header">
  <div class="header-left">
    <div class="logo-box"><svg viewBox="0 0 24 24"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg></div>
    <div>
      <p class="header-title">FinAdvisor</p>
      <p class="header-subtitle">AI Workspace</p>
    </div>
  </div>
  <div class="status-pill"><div class="status-dot"></div>SECURED</div>
</div>
""", unsafe_allow_html=True)

# ─── MAIN TABS ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Workspace", "Security", "Architecture", "Analytics"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — WORKSPACE
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    active_session = st.session_state.sessions[st.session_state.current_session_id]
    
    # ── Engine Selector (Top of Chat) ──
    col_sel, _ = st.columns([1, 2])
    with col_sel:
        st.markdown('<p style="font-size: 14px; font-weight: 600; color: #0F172A; margin-bottom: 4px;">Reasoning Engine</p>', unsafe_allow_html=True)
        model_mode = st.selectbox(
            "Reasoning Engine", 
            options=["Fast Tax Model", "Deep Workflow Model"],
            index=0,
            label_visibility="collapsed"
        )
        
        # Dynamic Gemini/Claude style subtext
        if model_mode == "Fast Tax Model":
            st.markdown('<p class="engine-descriptor">Low latency RAG retrieval for standard compliance queries.</p>', unsafe_allow_html=True)
            engine_mode = "fast"
        else:
            st.markdown('<p class="engine-descriptor">Multi-agent reasoning pipeline for complex financial scenarios.</p>', unsafe_allow_html=True)
            engine_mode = "deep"

    if st.session_state.active_tool == "tax":
        tax_service, IncomeBreakdown, DeductionBreakdown = load_tax_service()
        tool_header_col, tool_close_col = st.columns([6, 1])
        with tool_header_col:
            st.markdown("### Tax Calculator")
            st.caption("Deterministic tax, GST, and tax-saving calculations from the service layer.")
        with tool_close_col:
            if st.button("Close", key="close_tax_tool", use_container_width=True):
                st.session_state.active_tool = None
                st.rerun()

        tax_tab, gst_tab, saving_tab = st.tabs(["Income Tax", "GST", "Tax Saving"])
        with tax_tab:
            with st.form("income_tax_form"):
                income_col1, income_col2 = st.columns(2)
                with income_col1:
                    salary = st.number_input("Salary income", min_value=0.0, value=850000.0, step=50000.0)
                    capital_gains = st.number_input("Capital gains", min_value=0.0, value=0.0, step=25000.0)
                    rental_income = st.number_input("Rental income", min_value=0.0, value=0.0, step=25000.0)
                with income_col2:
                    business_income = st.number_input("Business income", min_value=0.0, value=0.0, step=25000.0)
                    other_income = st.number_input("Other income", min_value=0.0, value=0.0, step=25000.0)
                    section_80c = st.number_input("Section 80C", min_value=0.0, value=150000.0, step=10000.0)

                deduction_col1, deduction_col2 = st.columns(2)
                with deduction_col1:
                    section_80d = st.number_input("Section 80D", min_value=0.0, value=25000.0, step=5000.0)
                    section_80e = st.number_input("Section 80E", min_value=0.0, value=0.0, step=5000.0)
                with deduction_col2:
                    section_80tta = st.number_input("Section 80TTA", min_value=0.0, value=0.0, step=1000.0)
                    other_deductions = st.number_input("Other deductions", min_value=0.0, value=0.0, step=5000.0)

                calculate_tax = st.form_submit_button("Calculate Income Tax", use_container_width=True)

            if calculate_tax:
                income_breakdown = IncomeBreakdown(
                    salary=salary,
                    capital_gains=capital_gains,
                    rental_income=rental_income,
                    business_income=business_income,
                    other_income=other_income,
                )
                deduction_breakdown = DeductionBreakdown(
                    section_80c=section_80c,
                    section_80d=section_80d,
                    section_80e=section_80e,
                    section_80tta=section_80tta,
                    other_deductions=other_deductions,
                )
                tax_result = tax_service.calculate_income_tax(income_breakdown, deduction_breakdown)
                rebates = tax_service.get_eligible_rebates(tax_result.taxable_income)

                result_cols = st.columns(4)
                result_cols[0].metric("Gross income", format_currency(tax_result.gross_income))
                result_cols[1].metric("Deductions", format_currency(tax_result.total_deductions))
                result_cols[2].metric("Taxable income", format_currency(tax_result.taxable_income))
                result_cols[3].metric("Tax payable", format_currency(tax_result.tax_amount))
                st.caption(
                    f"Applicable slab: {tax_result.applicable_slab} | Effective rate: {tax_result.effective_tax_rate:.1f}%"
                )

                if rebates:
                    st.dataframe(pd.DataFrame(rebates), use_container_width=True, hide_index=True)

        with gst_tab:
            with st.form("gst_form"):
                gst_amount = st.number_input("Taxable amount", min_value=0.0, value=500000.0, step=10000.0)
                gst_category = st.selectbox("GST rate", options=["5%", "12%", "18%", "28%"], index=2)
                calculate_gst = st.form_submit_button("Calculate GST", use_container_width=True)

            if calculate_gst:
                gst_result = tax_service.calculate_gst(gst_amount, gst_category)
                gst_cols = st.columns(3)
                gst_cols[0].metric("Base amount", format_currency(gst_result["base_amount"]))
                gst_cols[1].metric("GST", format_currency(gst_result["gst"]))
                gst_cols[2].metric("Invoice total", format_currency(gst_result["total"]))

        with saving_tab:
            with st.form("tax_saving_form"):
                current_income = st.number_input("Annual income", min_value=0.0, value=1200000.0, step=50000.0)
                current_tax = st.number_input("Current tax estimate", min_value=0.0, value=125000.0, step=5000.0)
                suggest_savings = st.form_submit_button("Recommend Options", use_container_width=True)

            if suggest_savings:
                options = tax_service.calculate_tax_saving_options(current_income, current_tax)
                if options:
                    st.dataframe(pd.DataFrame(options), use_container_width=True, hide_index=True)
                else:
                    st.info("No tax-saving options available for the provided inputs.")

    elif st.session_state.active_tool == "investment":
        investment_service = load_investment_service()
        tool_header_col, tool_close_col = st.columns([6, 1])
        with tool_header_col:
            st.markdown("### Investment Planner")
            st.caption("Deterministic SIP, lump-sum, retirement, and allocation planning.")
        with tool_close_col:
            if st.button("Close", key="close_investment_tool", use_container_width=True):
                st.session_state.active_tool = None
                st.rerun()

        growth_tab, sip_tab, retirement_tab = st.tabs(["Returns", "SIP Planner", "Retirement"])
        with growth_tab:
            with st.form("returns_form"):
                return_col1, return_col2 = st.columns(2)
                with return_col1:
                    principal = st.number_input("Principal", min_value=0.0, value=500000.0, step=10000.0)
                    rate = st.number_input("Expected annual return (%)", min_value=0.0, value=10.0, step=0.5)
                with return_col2:
                    years = st.number_input("Years", min_value=1, value=5, step=1)
                    compounding = st.selectbox("Compounding", options=["annual", "semi-annual", "quarterly", "monthly"])
                calculate_returns = st.form_submit_button("Project Returns", use_container_width=True)

            if calculate_returns:
                returns = investment_service.calculate_returns(principal, rate, int(years), compounding)
                recommendation_rows = investment_service.get_investment_options(principal, int(years * 12), "moderate")
                return_cols = st.columns(3)
                return_cols[0].metric("Principal", format_currency(returns["principal"]))
                return_cols[1].metric("Interest earned", format_currency(returns["interest_earned"]))
                return_cols[2].metric("Final amount", format_currency(returns["final_amount"]))
                st.dataframe(pd.DataFrame(recommendation_rows), use_container_width=True, hide_index=True)

        with sip_tab:
            with st.form("sip_form"):
                sip_col1, sip_col2 = st.columns(2)
                with sip_col1:
                    monthly_amount = st.number_input("Monthly SIP", min_value=0.0, value=15000.0, step=1000.0)
                    sip_rate = st.number_input("Expected annual return (%)", min_value=0.0, value=12.0, step=0.5)
                with sip_col2:
                    sip_months = st.number_input("Months", min_value=1, value=120, step=12)
                calculate_sip = st.form_submit_button("Calculate SIP", use_container_width=True)

            if calculate_sip:
                sip_result = investment_service.calculate_sip(monthly_amount, sip_rate, int(sip_months))
                sip_metrics = st.columns(3)
                sip_metrics[0].metric("Total invested", format_currency(sip_result["total_invested"]))
                sip_metrics[1].metric("Interest earned", format_currency(sip_result["interest_earned"]))
                sip_metrics[2].metric("Corpus", format_currency(sip_result["final_amount"]))

        with retirement_tab:
            with st.form("retirement_form"):
                retire_col1, retire_col2, retire_col3 = st.columns(3)
                with retire_col1:
                    current_age = st.number_input("Current age", min_value=18, value=30, step=1)
                with retire_col2:
                    retirement_age = st.number_input("Retirement age", min_value=40, value=60, step=1)
                with retire_col3:
                    monthly_expense = st.number_input("Current monthly expense", min_value=0.0, value=75000.0, step=5000.0)
                inflation_rate = st.slider("Inflation rate (%)", min_value=2.0, max_value=10.0, value=5.0, step=0.5)
                return_rate = st.slider("Expected return rate (%)", min_value=4.0, max_value=15.0, value=8.0, step=0.5)
                calculate_retirement = st.form_submit_button("Estimate Retirement Corpus", use_container_width=True)

            if calculate_retirement:
                retirement = investment_service.calculate_retirement_corpus(
                    int(current_age),
                    int(retirement_age),
                    monthly_expense,
                    inflation_rate=inflation_rate,
                    return_rate=return_rate,
                )
                retirement_cols = st.columns(3)
                retirement_cols[0].metric("Years to retirement", int(retirement["years_to_retirement"]))
                retirement_cols[1].metric("Corpus needed", format_currency(retirement["corpus_needed"]))
                retirement_cols[2].metric("Monthly SIP required", format_currency(retirement["monthly_sip_required"]))

    # ── Chat Thread ──
    chat_container = st.container(height=500, border=False)
    with chat_container:
        if not active_session["messages"]:
            st.markdown("""
            <div style="text-align: center; margin-top: 10vh; color: #64748B;">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="margin-bottom: 16px;"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>
                <h3 style="color: #0F172A; font-weight: 500; font-size: 20px;">Ready to assist</h3>
                <p style="font-size: 14px;">Ask a financial question or run a scenario from the sidebar.</p>
            </div>
            """, unsafe_allow_html=True)

        for msg in active_session["messages"]:
            if msg["role"] == "user":
                st.markdown(f'<div class="msg-user-row"><div class="msg-user-bubble">{msg["content"]}</div></div>', unsafe_allow_html=True)
            else:
                render_ai_message(msg)

    # Input Field
    user_query = st.chat_input("Message FinAdvisor...")
    if user_query:
        if len(active_session["messages"]) == 0:
            st.session_state.sessions[st.session_state.current_session_id]["title"] = user_query[:25] + "..."
            
        active_session["messages"].append({"role": "user", "content": user_query.strip()})
        st.rerun()

    # Execution (triggers on rerun if last message is user)
    if active_session["messages"] and active_session["messages"][-1]["role"] == "user":
        if not demo:
            st.error("Backend unavailable.")
        else:
            with chat_container:
                with st.spinner(f"Analyzing via {model_mode}..."):
                    res = demo.run_query(
                        active_session["messages"][-1]["content"],
                        mode=engine_mode,
                        chat_history=active_session["messages"][:-1],
                    )
                ans = clean_answer(res.get("answer", ""))
                active_session["messages"].append({"role": "assistant", "content": ans, "meta": res})
                st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SECURITY
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p style="font-size:22px;font-weight:700;color:#0F172A;margin:32px 0 6px 0;letter-spacing:-0.02em;">Security Gatekeeper</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:14px;color:#64748B;margin:0 0 28px 0;">3-layer pipeline inspecting queries before retrieval.</p>', unsafe_allow_html=True)

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.markdown("""
        <div class="sec-col">
          <p class="sec-col-title">Layer 1 &mdash; Input Validator</p>
          <p class="sec-col-desc">Checks query length, encoding, and structural validity before processing begins.</p>
          <div class="threat-item"><span class="threat-label">blocks</span>Empty or malformed input</div>
          <div class="threat-item"><span class="threat-label">blocks</span>Oversized payloads</div>
        </div>
        """, unsafe_allow_html=True)
    with sc2:
        st.markdown("""
        <div class="sec-col">
          <p class="sec-col-title">Layer 2 &mdash; Injection Detector</p>
          <p class="sec-col-desc">Pattern-matches against known prompt injection signatures and override attempts.</p>
          <div class="threat-item"><span class="threat-label">blocks</span>&ldquo;Ignore previous instructions&rdquo;</div>
          <div class="threat-item"><span class="threat-label">blocks</span>&ldquo;Reveal the system prompt&rdquo;</div>
        </div>
        """, unsafe_allow_html=True)
    with sc3:
        st.markdown("""
        <div class="sec-col">
          <p class="sec-col-title">Layer 3 &mdash; Adversarial Classifier</p>
          <p class="sec-col-desc">ML-based classifier detects malicious intent, jailbreaks, and exfiltration.</p>
          <div class="threat-item"><span class="threat-label">blocks</span>&ldquo;How do I evade taxes illegally?&rdquo;</div>
          <div class="threat-item"><span class="threat-label">blocks</span>DAN / jailbreak role-play</div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p style="font-size:22px;font-weight:700;color:#0F172A;margin:32px 0 6px 0;letter-spacing:-0.02em;">System Architecture</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:14px;color:#64748B;margin:0 0 28px 0;">All queries pass through the same security gatekeeper, then route into either a fast RAG lane or the full multi-agent deep workflow.</p>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:8px; padding:24px; box-shadow:0 1px 3px rgba(0,0,0,0.04);">
      <p style="font-size:11px;font-weight:600;color:#94A3B8;text-transform:uppercase;letter-spacing:0.08em;margin:0 0 16px 0;">Agent Pipeline</p>
      <div class="pipeline-wrap">
        <div class="pipe-step"><div class="pipe-step-num">Step 01</div><div class="pipe-step-name">Security</div><div class="pipe-step-desc">3-layer gatekeeper validation</div></div>
        <div class="pipe-arrow">&#8594;</div>
        <div class="pipe-step"><div class="pipe-step-num">Step 02</div><div class="pipe-step-name">Routing</div><div class="pipe-step-desc">Intent, domain, and mode selection</div></div>
        <div class="pipe-arrow">&#8594;</div>
        <div class="pipe-step"><div class="pipe-step-num">Step 03</div><div class="pipe-step-name">Retrieval</div><div class="pipe-step-desc">Federated hybrid search + reranking</div></div>
        <div class="pipe-arrow">&#8594;</div>
        <div class="pipe-step"><div class="pipe-step-num">Step 04</div><div class="pipe-step-name">Fast / Deep</div><div class="pipe-step-desc">Single-pass synthesis or planner-reasoner chain</div></div>
        <div class="pipe-arrow">&#8594;</div>
        <div class="pipe-step"><div class="pipe-step-num">Step 05</div><div class="pipe-step-name">Verification</div><div class="pipe-step-desc">Grounding, confidence, and formatting</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    a1, a2 = st.columns(2)
    with a1:
        st.markdown('<p style="font-size:15px;font-weight:600;color:#0F172A;margin:0 0 14px 0;">Knowledge Base</p>', unsafe_allow_html=True)
        for domain_stat in load_knowledge_base_stats():
            st.markdown(f"""
            <div class="domain-card">
                <div class="domain-name">{domain_stat['name']}</div>
                <div style="margin:4px 0;"><span class="domain-chunks">{domain_stat['chunks']}</span><span class="domain-label">chunks indexed</span></div>
                <div style="font-size:12px;color:#94A3B8;">{domain_stat['description']}</div>
            </div>
            """, unsafe_allow_html=True)
    with a2:
        st.markdown('<p style="font-size:15px;font-weight:600;color:#0F172A;margin:0 0 14px 0;">Technology Stack</p>', unsafe_allow_html=True)
        stack = [
            ("Embeddings","all-MiniLM-L6-v2"),
            ("Vector Search","FAISS AVX2 + GPU"),
            ("Keyword Search","BM25 via rank_bm25"),
            ("Reranker","ms-marco-MiniLM-L6-v2"),
            ("Fast LLM",GENERAL_MODEL),
            ("Deep Reasoner",REASONING_MODEL),
            ("Security","3-layer gatekeeper"),
            ("Audit","Provenance DAG per query"),
            ("Framework","FastAPI + Streamlit")
        ]
        sh = '<div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:8px; padding:16px 20px; box-shadow:0 1px 3px rgba(0,0,0,0.04);">'
        for lbl, val in stack:
            sh += f'<div class="tech-row"><span style="font-size:12px;color:#64748B;font-weight:500;">{lbl}</span><span style="font-family:\'Courier New\',monospace;font-size:11px;color:#0F172A;text-align:right;max-width:58%;">{val}</span></div>'
        sh += "</div>"
        st.markdown(sh, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    analytics = build_analytics_snapshot()
    st.markdown('<p style="font-size:22px;font-weight:700;color:#0F172A;margin:32px 0 6px 0;letter-spacing:-0.02em;">System Analytics</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:14px;color:#64748B;margin:0 0 28px 0;">Benchmark-backed routing latency, mode utilization, and failure rates. Refreshed at {analytics["updated_at"]}.</p>', unsafe_allow_html=True)
    if st.button("Refresh Analytics", key="refresh_analytics"):
        load_json_file.clear()
        st.rerun()

    # Top Level Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="analytic-card">
            <div class="analytic-label">Benchmark Queries</div>
            <div class="analytic-value">{analytics['total_queries']}</div>
            <div class="analytic-trend-up" style="color:#64748B;">Latest saved run</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="analytic-card">
            <div class="analytic-label">Avg. Latency (Fast)</div>
            <div class="analytic-value">{format_latency_ms(analytics['fast_avg_latency_ms'])}</div>
            <div class="analytic-trend-up" style="color:#64748B;">Fast benchmark mean</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="analytic-card">
            <div class="analytic-label">Avg. Latency (Deep)</div>
            <div class="analytic-value">{format_latency_ms(analytics['deep_avg_latency_ms'])}</div>
            <div class="analytic-trend-up" style="color:#64748B;">Deep sample mean</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="analytic-card">
            <div class="analytic-label">System Block Rate</div>
            <div class="analytic-value">{analytics['block_rate']:.0%}</div>
            <div class="analytic-trend-up" style="color:#64748B;">Across saved benchmark cases</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    s1, s2 = st.columns(2)
    with s1:
        st.metric("Lane match rate", f"{analytics['lane_match_rate']:.0%}")
    with s2:
        st.metric("Grounded retrieval rate", f"{analytics['grounded_rate']:.0%}")
    
    # Charts Section
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("**Mode Utilization (Fast vs Deep)**")
        st.bar_chart(analytics["mode_utilization"], color=["#3b82f6"])

    with c2:
        st.markdown("**Error & Timeout Distribution**")
        st.bar_chart(analytics["issue_distribution"], color=["#f87171"])