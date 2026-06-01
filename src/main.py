"""医疗健康智能助手 — Streamlit Web UI (robust version, no cache deadlock)."""

import streamlit as st
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.safety_agent import SafetyAgent
from src.agents.router_agent import RouterAgent
from src.agents.business_agent import BusinessAgent
from src.agents.reflection_agent import ReflectionAgent
from src.config import LLM_PROVIDER, get_llm_config


# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="MedChoice AI · 医疗健康智能助手",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
if "css_loaded" not in st.session_state:
    st.markdown("""<style>/* === CSS Variables & Fonts === */
:root {
    --bg-deep: #080c17;
    --bg-mid: #0f1729;
    --bg-card: rgba(255, 255, 255, 0.025);
    --border-subtle: rgba(255, 255, 255, 0.06);
    --border-hover: rgba(34, 211, 238, 0.25);
    --accent: #22d3ee;
    --accent-dim: rgba(34, 211, 238, 0.15);
    --accent2: #818cf8;
    --accent2-dim: rgba(129, 140, 248, 0.15);
    --text-primary: #e2e8f0;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --success: #34d399;
    --warning: #fbbf24;
    --danger: #f87171;
}

/* === Base === */
html, body, .stApp, [class*="css"] {
    font-family: system-ui, -apple-system, "Segoe UI", "Helvetica Neue", sans-serif !important;
    color: var(--text-primary);
}
.stApp {
    background-color: var(--bg-deep);
    background-image:
        radial-gradient(ellipse at 20% 50%, rgba(34, 211, 238, 0.03) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 20%, rgba(129, 140, 248, 0.04) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 80%, rgba(34, 211, 238, 0.02) 0%, transparent 50%);
}
/* === Scrollbar === */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(255, 255, 255, 0.15); }
/* === AI Status Indicator === */
.ai-status {
    position: fixed;
    top: 20px;
    right: 28px;
    z-index: 9999;
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    background: rgba(15, 23, 41, 0.85);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 24px;
    font-size: 13px;
    font-weight: 500;
    color: var(--text-secondary);
    letter-spacing: 0.01em;
}
.ai-status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--success);
    display: inline-block;
    box-shadow: 0 0 8px rgba(52, 211, 153, 0.6), 0 0 20px rgba(52, 211, 153, 0.3);
    animation: breathe 2s ease-in-out infinite;
}



/* === Glass Card === */
.glass-card {
    background: rgba(255, 255, 255, 0.025);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 20px;
    padding: 24px;
    margin: 16px 0;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2);
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
}
.glass-card:hover {
    border-color: var(--border-hover);
    box-shadow: 0 4px 32px rgba(34, 211, 238, 0.08), 0 0 0 1px rgba(34, 211, 238, 0.06);
    transform: translateY(-1px);
}

/* === Gradient Border Card === */
.gradient-card {
    position: relative;
    background: rgba(255, 255, 255, 0.025);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 20px;
    padding: 20px 24px;
    margin: 12px 0;
}
.gradient-card::before {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: 20px;
    padding: 1px;
    background: linear-gradient(135deg, rgba(34, 211, 238, 0.15), rgba(129, 140, 248, 0.15), transparent 60%);
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor;
    mask-composite: exclude;
    pointer-events: none;
}

/* === Tab Radio === */
div[role="radiogroup"] {
    display: flex;
    gap: 6px;
    padding: 6px;
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    margin-bottom: 24px;
}
div[role="radiogroup"] label {
    flex: 1;
    padding: 10px 16px !important;
    border-radius: 12px !important;
    text-align: center;
    font-weight: 500;
    font-size: 14px;
    color: var(--text-secondary) !important;
    cursor: pointer;
    transition: all 0.3s ease;
    margin: 0 !important;
    border: none !important;
}
div[role="radiogroup"] label:hover {
    color: var(--text-primary) !important;
    background: rgba(255, 255, 255, 0.03);
}
div[role="radiogroup"] label[data-selected="true"],
div[role="radiogroup"] label[aria-checked="true"] {
    background: linear-gradient(135deg, rgba(34, 211, 238, 0.2), rgba(129, 140, 248, 0.2)) !important;
    color: var(--accent) !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 12px rgba(34, 211, 238, 0.15);
}
div[role="radiogroup"] input[type="radio"] {
    display: none !important;
}

/* === File Uploader === */
[data-testid="stFileUploader"] {
    background: rgba(255, 255, 255, 0.015) !important;
    border: 2px dashed rgba(34, 211, 238, 0.2) !important;
    border-radius: 16px !important;
    padding: 16px !important;
    transition: all 0.3s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(34, 211, 238, 0.5) !important;
    background: rgba(34, 211, 238, 0.03) !important;
    box-shadow: 0 0 24px rgba(34, 211, 238, 0.06) !important;
}
[data-testid="stFileUploader"] p { color: var(--text-secondary) !important; }
[data-testid="stFileUploader"] small { color: var(--text-muted) !important; }

/* === Sidebar === */
[data-testid="stSidebar"] {
    background: rgba(15, 23, 41, 0.95) !important;
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
}
[data-testid="stSidebar"] * { color: var(--text-secondary) !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] label {
    color: var(--text-primary) !important;
    font-family: system-ui, -apple-system, "Segoe UI", "Helvetica Neue", sans-serif !important;
}
[data-testid="stSidebar"] input, [data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] select, [data-testid="stSidebar"] [data-baseweb="select"],
[data-testid="stSidebar"] .stNumberInput input {
    background: rgba(255, 255, 255, 0.04) !important;
    color: var(--text-primary) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 10px !important;
}
[data-testid="stSidebar"] input:focus, [data-testid="stSidebar"] textarea:focus {
    border-color: rgba(34, 211, 238, 0.4) !important;
    box-shadow: 0 0 0 2px rgba(34, 211, 238, 0.1) !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, rgba(34, 211, 238, 0.2), rgba(129, 140, 248, 0.2)) !important;
    border: 1px solid rgba(34, 211, 238, 0.15) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(135deg, rgba(34, 211, 238, 0.35), rgba(129, 140, 248, 0.35)) !important;
    border-color: rgba(34, 211, 238, 0.4) !important;
}
[data-testid="stSidebar"] [data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg, rgba(52, 211, 153, 0.25), rgba(34, 211, 238, 0.25)) !important;
    border: 1px solid rgba(52, 211, 153, 0.3) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] [data-testid="stFormSubmitButton"] button:hover {
    background: linear-gradient(135deg, rgba(52, 211, 153, 0.4), rgba(34, 211, 238, 0.4)) !important;
    box-shadow: 0 4px 20px rgba(52, 211, 153, 0.2) !important;
}

/* === Spinner === */
.stSpinner > div {
    border-color: var(--accent) !important;
    border-top-color: transparent !important;
}

/* === Loading Dots === */
.loading-dots {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    padding: 24px;
}
.loading-dots span {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--accent);
    display: inline-block;
    animation: bounce 1.4s ease-in-out infinite both;
}
.loading-dots span:nth-child(1) { animation-delay: -0.32s; }
.loading-dots span:nth-child(2) { animation-delay: -0.16s; }
.loading-text {
    text-align: center;
    color: var(--text-muted);
    font-size: 13px;
    font-weight: 500;
    margin-top: 8px;
    letter-spacing: 0.04em;
}

/* === Disclaimer === */
.disclaimer-box {
    background: rgba(251, 191, 36, 0.06);
    border: 1px solid rgba(251, 191, 36, 0.2);
    border-left: 4px solid var(--warning);
    border-radius: 12px;
    padding: 12px 18px;
    margin: 16px 0;
    font-weight: 500;
    font-size: 13px;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
}

/* === Expander === */
.streamlit-expanderHeader {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 12px !important;
    color: var(--text-secondary) !important;
    font-family: system-ui, -apple-system, "Segoe UI", "Helvetica Neue", sans-serif !important;
    font-weight: 500 !important;
}
.streamlit-expanderHeader:hover {
    border-color: rgba(34, 211, 238, 0.2) !important;
}

/* === Buttons === */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, rgba(34, 211, 238, 0.2), rgba(129, 140, 248, 0.2)) !important;
    border: 1px solid rgba(34, 211, 238, 0.2) !important;
    color: var(--accent) !important;
    font-weight: 600 !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, rgba(34, 211, 238, 0.35), rgba(129, 140, 248, 0.35)) !important;
    box-shadow: 0 4px 20px rgba(34, 211, 238, 0.15) !important;
}

/* === Text Area === */
.stTextArea textarea, .stTextInput input {
    background: rgba(255, 255, 255, 0.03) !important;
    color: var(--text-primary) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    font-family: system-ui, -apple-system, "Segoe UI", "Helvetica Neue", sans-serif !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: rgba(34, 211, 238, 0.3) !important;
    box-shadow: 0 0 0 2px rgba(34, 211, 238, 0.05) !important;
}

/* === Success / Info / Warning Boxes === */
.stAlert {
    border-radius: 12px !important;
    border: none !important;
}
div[data-testid="stAlert"] {
    border-radius: 12px !important;
    padding: 12px 16px !important;
}

/* === Footer === */
.app-footer {
    text-align: center;
    padding: 20px 0 12px;
    color: var(--text-muted);
    font-size: 12px;
    letter-spacing: 0.03em;
    border-top: 1px solid rgba(255, 255, 255, 0.04);
    margin-top: 32px;
}

/* === Keyframes === */
@keyframes breathe {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.85); }
}
@keyframes bounce {
    0%, 80%, 100% { transform: scale(0); }
    40% { transform: scale(1); }
}
</style>""", unsafe_allow_html=True)
    st.session_state.css_loaded = True


# ── Agent init (session state — runs ONCE, never re-executes) ─
def _init_agents():
    return {
        "safety": SafetyAgent(),
        "router": RouterAgent(),
        "business": BusinessAgent(),
        "reflection": ReflectionAgent(max_retries=2),
    }


if "agents" not in st.session_state:
    st.session_state.agents = _init_agents()

AGENTS = st.session_state.agents
USER_ID = "default"

# Chat history for multi-turn conversation
# Chat history per scenario (isolated across tabs)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def get_llm_label() -> str:
    cfg = get_llm_config()
    return f"{LLM_PROVIDER}/{cfg['model']}"


# Buffer for auto-fill buttons (avoids Streamlit "cannot modify widget" error)
for _k in ["exam_buf", "drug_buf", "rpt_buf", "dept_buf"]:
    if _k not in st.session_state:
        st.session_state[_k] = ""

# ── UI ────────────────────────────────────────────────────────
st.title("🏥 医疗健康智能助手")
st.caption("MedChoice — 体检选择 · 药品对比 · 报告解读 · 科室推荐")

# AI status badge
st.markdown('<div class="ai-status"><span class="ai-status-dot"></span> 🤖 MedChoice AI · 在线</div>', unsafe_allow_html=True)

st.markdown(f'<div class="disclaimer-box">⚠️ <b>重要提示：</b>本工具为健康决策辅助参考，<b>不能替代医生诊断</b>。所有分析仅供您与医生沟通时参考，请以专业医疗机构意见为准。使用 {get_llm_label()} 驱动。</div>', unsafe_allow_html=True)

# 新手欢迎引导（仅首次打开时显示）
if "welcome_shown" not in st.session_state:
    st.session_state.welcome_shown = True



# Sidebar: profile
profile = AGENTS["business"].get_profile(USER_ID)
profile_path = Path("data/user_profiles/default.json")
with st.sidebar:
    st.title("👤 个人健康画像")
    # Cache mtime in session_state — only re-stat when profile is saved
    if "profile_mtime_str" not in st.session_state:
        if profile_path.exists():
            import os as _os
            from datetime import datetime
            mtime = _os.path.getmtime(profile_path)
            st.session_state.profile_mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        else:
            st.session_state.profile_mtime_str = ""
    if st.session_state.profile_mtime_str:
        st.caption(f"✅ 画像已加载 (上次保存: {st.session_state.profile_mtime_str})")
    else:
        st.caption("📝 首次使用，请填写并保存画像")

    with st.form("profile_form"):
        age = st.number_input("年龄", 0, 120, profile.age)
        gender = st.selectbox(
            "性别", ["", "男", "女"],
            index=(0 if not profile.gender else 1 if profile.gender == "男" else 2),
        )
        occupation = st.text_input("职业", profile.occupation)
        budget = st.number_input("体检预算（元）", 0, 100000, profile.budget, 100)
        family_history = st.text_area("家族病史（选填）", profile.family_history, height=60)
        chronic = st.text_area("已有慢性病（选填）", profile.chronic_conditions, height=60)
        if st.form_submit_button("💾 保存画像", use_container_width=True):
            AGENTS["business"].update_profile(
                USER_ID, age=age, gender=gender, occupation=occupation,
                budget=budget, family_history=family_history, chronic_conditions=chronic,
            )
            # Refresh cached mtime
            from datetime import datetime
            st.session_state.profile_mtime_str = datetime.now().strftime("%Y-%m-%d %H:%M")
            st.success("画像已保存！")
    st.divider()
    st.caption("🔒 隐私说明：您的健康画像数据仅存储在本地，不会被上传或分享给第三方。您可以随时修改或删除。")

def _strip_markdown(text: str) -> str:
    """Strip common markdown formatting for clean preview display."""
    import re
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(r"\|", " ", text)
    text = re.sub(r"#{1,6}\s*", "", text)
    text = re.sub(r"[-*+]\s+", "", text)
    text = re.sub(r"\n{2,}", " ", text)
    return text.strip()

if st.session_state.chat_history:
    with st.expander("💬 对话历史 (" + str(len(st.session_state.chat_history)//2) + " 轮)", expanded=False):
        for i, msg in enumerate(st.session_state.chat_history[-6:]):
            if msg["role"] == "user":
                st.caption("👤 你: " + _strip_markdown(msg["content"][:150]))
            else:
                preview = _strip_markdown(msg["content"][:150])
                st.caption("🤖 助手: " + preview + ("..." if len(msg["content"]) > 150 else ""))
        if st.button("🗑️ 清除对话历史", key="clear_history"):
            st.session_state.chat_history = []
            st.rerun()



# Tabs — key= auto-manages session_state, no double-click bug
TAB_LABELS = ["🔍 体检套餐选择", "💊 药品对比", "📋 体检报告解读", "🏥 就医科室推荐"]

active_tab = st.radio(
    "导航",
    TAB_LABELS,
    horizontal=True,
    label_visibility="collapsed",
    key="active_tab",
)

def _stream_tab(text, uploaded_file=None, uploaded_file2=None, tab_label=""):
    """Shared streaming tab handler. Renders progress + streamed output."""
    t0 = time.time()
    history = st.session_state.chat_history
    try:
        # Phase 1+2: Safety + Routing (not streamed)
        # 场景专属加载提示
        scenario_messages = {
            "physical_exam": "正在分析您的健康画像，匹配体检套餐...",
            "drug_compare": "正在检索药品信息，准备对比分析...",
            "report_reading": "正在解读体检指标，请稍候...",
            "department_recommendation": "正在分析症状，匹配就诊科室...",
        }
        spinner_msg = scenario_messages.get(tab_label, "正在分析您的问题...")
        with st.spinner(spinner_msg):
            safety_result = AGENTS["safety"].check(text)
            if not safety_result["safe"]:
                st.error(safety_result["guidance"])
                return

            route_result = AGENTS["router"].route(text)
            # Fallback: if routing fails but user uploaded a file, use tab_label as hint
            if route_result["scenario"] == "unknown" and uploaded_file is not None and tab_label:
                route_result = {"scenario": tab_label, "confidence": 0.5}
            elif route_result["scenario"] == "unknown":
                st.info(RouterAgent.get_guidance(route_result["scenario"]))
                return

        # Phase 3: Business (streamed, with history)
        file_path = None
        file_path2 = None
        if uploaded_file is not None:
            tmp_dir = Path("data/temp")
            tmp_dir.mkdir(parents=True, exist_ok=True)
            file_path = str(tmp_dir / uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            output_placeholder = st.empty()
            output_placeholder.info("📄 文件已接收，正在解析内容...")
            time.sleep(0.3)

        if uploaded_file2 is not None:
            tmp_dir = Path("data/temp")
            tmp_dir.mkdir(parents=True, exist_ok=True)
            file_path2 = str(tmp_dir / ("b_" + uploaded_file2.name))
            with open(file_path2, "wb") as f:
                f.write(uploaded_file2.getbuffer())

        output_placeholder = st.empty()
        output_placeholder.markdown('<div class="loading-dots"><span></span><span></span><span></span></div><div class="loading-text">AI 正在分析...</div>', unsafe_allow_html=True)
        streamed_text = ""
        for chunk in AGENTS["business"].run_stream(
            route_result["scenario"], text, USER_ID, file_path, history, file_path2
        ):
            streamed_text += chunk
            output_placeholder.markdown(streamed_text + " ▌")

        # Phase 4: Reflection (runs silently in background)
        reflection_note = ""
        if streamed_text:
            reflection_result = AGENTS["reflection"].verify(streamed_text)
            if not reflection_result["pass"]:
                issues = []
                has_consistency_issue = False
                for name, check in reflection_result.get("checks", {}).items():
                    if not check.get("pass", True):
                        issues.append(f"[{name}] {check.get('issue', '')}")
                        if name == "consistency":
                            has_consistency_issue = True
                if issues:
                    if has_consistency_issue:
                        reflection_note = "⚠️ *注意：当前知识库数据可能不足以完全支撑本次推荐，以下结果仅供参考。*\n\n"
                    else:
                        with st.spinner("正在优化回复..."):
                            retry = AGENTS["business"].run(
                                route_result["scenario"],
                                f"请修正以下问题后重新输出：\n" + "\n".join(issues),
                                USER_ID,
                                file_path,
                                history,
                            )
                        streamed_text = retry.get("output", streamed_text)
            else:
                reflection_note = "✅ *内容已通过自检 (边界/一致性/安全性/免责声明/完整性)*\n\n"

        # Final output
        elapsed = time.time() - t0
        st.success(f"完成 ({elapsed:.1f}s)")
        output_placeholder.markdown(reflection_note + streamed_text)

        # Save to history (tagged with scenario for cross-tab isolation)
        st.session_state.chat_history.append(
            {"role": "user", "content": text, "scenario": route_result["scenario"]}
        )
        st.session_state.chat_history.append(
            {"role": "assistant", "content": streamed_text, "scenario": route_result["scenario"]}
        )
        # Keep only last 10 exchanges
        if len(st.session_state.chat_history) > 20:
            st.session_state.chat_history = st.session_state.chat_history[-20:]

    except Exception as e:
        st.error("😥 分析时遇到问题，请稍后重试。常见原因：网络连接不稳定或 API 配置有误。您可以在终端查看详细错误信息。")
        # Log technical details to console only, not exposed to user
        print(f"[ERROR] tab={tab_label} | {e}\n{traceback.format_exc()}")



if active_tab == TAB_LABELS[0]:
    st.subheader("🔍 体检套餐选择")
    st.caption("📋 基于您的健康画像（年龄/性别/职业/预算）匹配推荐")
    profile = AGENTS["business"].get_profile(USER_ID)
    if not profile.is_complete():
        st.info("👈 左侧「个人健康画像」还未填写，填写后推荐会更精准", icon="💡")
    if st.session_state.get("exam_buf"):
        st.session_state["exam"] = st.session_state["exam_buf"]
        st.session_state["exam_buf"] = ""
    text1 = st.text_area("描述你的需求", placeholder="例如：帮我推荐适合的体检套餐，或我想对比两款体检套餐的区别...", key="exam")
    if profile.is_complete():
        st.markdown(f'<div class="gradient-card" style="font-size:13px;color:var(--text-secondary);">💡 <b>AI 建议：</b>已读取您的健康画像（{profile.gender}，{profile.age}岁，预算{profile.budget}元），将智能匹配最适合的体检套餐方案</div>', unsafe_allow_html=True)
    # 快速示例问题
    with st.container():
        st.caption("💡 试试这样问（点击自动填入）：")
        ec1, ec2, ec3 = st.columns(3)
        if ec1.button("📌 我30岁男预算1000", key="ex1", use_container_width=True):
            st.session_state["exam_buf"] = "我30岁男性，预算1000元，推荐什么体检套餐？"
            st.rerun()
        if ec2.button("📌 对比两种套餐区别", key="ex2", use_container_width=True):
            st.session_state["exam_buf"] = "对比基础体检套餐和精英体检套餐的区别"
            st.rerun()
        if ec3.button("📌 中老年人适合什么", key="ex3", use_container_width=True):
            st.session_state["exam_buf"] = "我55岁，有高血压，适合什么体检套餐？"
            st.rerun()
    if st.button("开始分析", key="btn1", type="primary"):
        if not text1.strip():
            st.warning("请输入需求描述")
        else:
            _stream_tab(text1.strip(), tab_label="physical_exam")

elif active_tab == TAB_LABELS[1]:
    st.subheader("💊 药品/保健品对比")
    st.caption("💊 输入两种药品名称即可对比。也可分别上传两种药的说明书（拍照/PDF/TXT），系统自动识别文字用于精准对比。")
    if st.session_state.get("drug_buf"):
        st.session_state["drug"] = st.session_state["drug_buf"]
        st.session_state["drug_buf"] = ""
    text2 = st.text_area("描述要对比的药品", placeholder="例如：布洛芬和对乙酰氨基酚有什么区别？哪个副作用更小？", key="drug")
    # Two uploaders for drug A and drug B
    dc_u1, dc_u2 = st.columns(2)
    with dc_u1:
        dfile1 = st.file_uploader("📄 药品A说明书（可选）", type=["pdf", "txt", "jpg", "jpeg", "png"], key="df1")
        if dfile1 is not None:
            st.caption(f"✅ {dfile1.name} ({dfile1.size/1024:.0f}KB)")
    with dc_u2:
        dfile2 = st.file_uploader("📄 药品B说明书（可选）", type=["pdf", "txt", "jpg", "jpeg", "png"], key="df2")
        if dfile2 is not None:
            st.caption(f"✅ {dfile2.name} ({dfile2.size/1024:.0f}KB)")
    st.markdown('<div class="gradient-card" style="font-size:13px;color:var(--text-secondary);">💡 <b>提示：</b>只需输入两种药品名称即可对比，上传说明书为可选项。AI 将从成分、适应症、副作用、禁忌等维度进行结构化对比。</div>', unsafe_allow_html=True)
    # 快速示例问题
    with st.container():
        st.caption("💡 试试这样问（点击自动填入）：")
        dc1, dc2, dc3 = st.columns(3)
        if dc1.button("📌 布洛芬 vs 对乙酰氨基酚", key="ddx1", use_container_width=True):
            st.session_state["drug_buf"] = "布洛芬和对乙酰氨基酚有什么区别？哪个副作用更小？"
            st.rerun()
        if dc2.button("📌 阿莫西林和头孢区别", key="ddx2", use_container_width=True):
            st.session_state["drug_buf"] = "阿莫西林和头孢有什么区别？"
            st.rerun()
        if dc3.button("📌 维生素C和维生素E", key="ddx3", use_container_width=True):
            st.session_state["drug_buf"] = "维生素C和维生素E可以一起吃吗？各有什么作用？"
            st.rerun()
    if st.button("开始对比", key="btn2", type="primary"):
        if not text2.strip() and dfile1 is None and dfile2 is None:
            st.warning("请输入药品名称或上传说明书")
        else:
            if text2.strip():
                msg = text2.strip()
            else:
                names = []
                if dfile1:
                    names.append(dfile1.name.rsplit(".", 1)[0])
                if dfile2:
                    names.append(dfile2.name.rsplit(".", 1)[0])
                if names:
                    msg = f"请对比以下药品：{' 和 '.join(names)}。列出成分、适应症、不良反应和注意事项。"
                else:
                    msg = "请对比药品信息"
            _stream_tab(msg, dfile1, dfile2, tab_label="drug_compare")

elif active_tab == TAB_LABELS[2]:
    st.subheader("📋 体检报告解读")
    st.caption("📋 结合您的健康画像（年龄/性别/慢性病史）辅助解读")
    profile = AGENTS["business"].get_profile(USER_ID)
    if not profile.is_complete():
        st.info("👈 左侧「个人健康画像」可补充年龄/性别等信息，帮助解读更精准", icon="💡")
    if st.session_state.get("rpt_buf"):
        st.session_state["rpt"] = st.session_state["rpt_buf"]
        st.session_state["rpt_buf"] = ""
    text3 = st.text_area("描述异常指标或健康疑问", placeholder="例如：谷丙转氨酶65偏高什么意思？需要去看医生吗？挂什么科？", key="rpt")
    rfile = st.file_uploader("上传体检报告（可选，支持 PDF/JPG/PNG/TXT，单文件≤10MB）", type=["pdf", "txt", "jpg", "jpeg", "png"], key="rf")
    if rfile is not None:
        st.caption(f"✅ 已接收: {rfile.name} ({rfile.size/1024:.0f}KB)")
    if profile.is_complete():
        st.markdown(f'<div class="gradient-card" style="font-size:13px;color:var(--text-secondary);">💡 <b>AI 建议：</b>已结合您的年龄（{profile.age}岁）和慢性病史进行个性化解读，异常指标将按偏离程度分级提示</div>', unsafe_allow_html=True)
    # 快速示例问题
    with st.container():
        st.caption("💡 试试这样问（点击自动填入）：")
        rc1, rc2, rc3 = st.columns(3)
        if rc1.button("📌 谷丙转氨酶65偏高", key="rx1", use_container_width=True):
            st.session_state["rpt_buf"] = "谷丙转氨酶65偏高是什么意思？需要看医生吗？挂什么科？"
            st.rerun()
        if rc2.button("📌 尿酸高了怎么办", key="rx2", use_container_width=True):
            st.session_state["rpt_buf"] = "尿酸450偏高，需要注意什么？"
            st.rerun()
        if rc3.button("📌 血脂四项怎么看", key="rx3", use_container_width=True):
            st.session_state["rpt_buf"] = "我的血脂报告显示总胆固醇和甘油三酯偏高，是什么意思？"
            st.rerun()
    if st.button("开始解读", key="btn3", type="primary"):
        msg = text3.strip() or "请解读上传的体检报告"
        _stream_tab(msg, rfile, tab_label="report_reading")

elif active_tab == TAB_LABELS[3]:
    st.subheader("🏥 就医科室推荐")
    st.caption("🏥 结合您的健康画像（年龄/性别/慢性病史）推荐就诊科室，画像有助于更精准的科室匹配")
    if st.session_state.get("dept_buf"):
        st.session_state["dept"] = st.session_state["dept_buf"]
        st.session_state["dept_buf"] = ""
    text4 = st.text_area("描述你的症状", placeholder="例如：最近经常头痛胸闷，应该挂什么科？需要提前准备什么？", key="dept")
    st.markdown('<div class="gradient-card" style="font-size:13px;color:var(--text-secondary);">💡 <b>AI 建议：</b>请详细描述症状部位、持续时间、伴随症状，AI 将基于医学知识库推荐最匹配的科室</div>', unsafe_allow_html=True)
    # 快速示例问题
    with st.container():
        st.caption("💡 试试这样问（点击自动填入）：")
        dc1, dc2, dc3 = st.columns(3)
        if dc1.button("📌 头痛胸闷挂什么科", key="ddx_dept1", use_container_width=True):
            st.session_state["dept_buf"] = "最近经常头痛胸闷，应该挂什么科？"
            st.rerun()
        if dc2.button("📌 胃痛应该看什么科", key="ddx_dept2", use_container_width=True):
            st.session_state["dept_buf"] = "经常胃痛胃胀，应该挂什么科？需要提前做什么检查？"
            st.rerun()
        if dc3.button("📌 体检后哪些指标要复查", key="ddx_dept3", use_container_width=True):
            st.session_state["dept_buf"] = "体检发现有几项指标异常，应该去哪个科室复查？"
            st.rerun()
    if st.button("开始推荐", key="btn4", type="primary"):
        if not text4.strip():
            st.warning("请描述你的症状")
        else:
            _stream_tab(text4.strip(), tab_label="department_recommendation")

st.divider()
st.markdown('<div class="app-footer"><span>🏥 <b>MedChoice</b> · 健康决策辅助 · 结果仅供参考 · 如有不适请及时就医</span></div>', unsafe_allow_html=True)
