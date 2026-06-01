"""MedChoice Gradio App — secondary interface, 4 scenarios with file upload."""

import sys
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr
from src.agents.safety_agent import SafetyAgent
from src.agents.router_agent import RouterAgent
from src.agents.business_agent import BusinessAgent
from src.agents.reflection_agent import ReflectionAgent

from src.config import LLM_PROVIDER, get_llm_config

cfg = get_llm_config()
LLM_LABEL = f"{LLM_PROVIDER}/{cfg['model']}"

safety = SafetyAgent()
router = RouterAgent()
business = BusinessAgent()
reflection = ReflectionAgent(max_retries=2)
USER_ID = "gradio_user"


def _save_upload(uploaded_file) -> str | None:
    """Save an uploaded file to temp dir. Returns file path or None."""
    if uploaded_file is None:
        return None
    tmp_dir = Path("data/temp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fname = Path(uploaded_file).name if isinstance(uploaded_file, str) else getattr(uploaded_file, "name", "upload")
    file_path = str(tmp_dir / fname)
    if isinstance(uploaded_file, str):
        from shutil import copyfile
        copyfile(uploaded_file, file_path)
    else:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read() if hasattr(uploaded_file, "read") else open(uploaded_file, "rb").read())
    return file_path


def run_pipeline(user_input, uploaded_file=None, uploaded_file2=None):
    """Execute the full 4-agent pipeline."""
    safety_result = safety.check(user_input)
    if not safety_result["safe"]:
        return safety_result["guidance"], "blocked"

    route_result = router.route(user_input)
    scenario = route_result["scenario"]
    if scenario == "unknown":
        return RouterAgent.get_guidance(scenario), "unknown"

    file_path = _save_upload(uploaded_file)
    file_path2 = _save_upload(uploaded_file2)

    business_result = business.run(scenario, user_input, USER_ID, file_path, uploaded_file2=file_path2)
    output = business_result.get("output", "")

    if output:
        ref_result = reflection.verify(output)
        if not ref_result["pass"]:
            issues = []
            for name, check in ref_result.get("checks", {}).items():
                if not check.get("pass", True):
                    issues.append(f"[{name}] {check.get('issue', '')}")
            if issues:
                retry = business.run(scenario, f"修正：\n" + "\n".join(issues), USER_ID, file_path, uploaded_file2=file_path2)
                output = retry.get("output", output)

    return output, scenario


def _timed(fn, *args):
    t0 = time.time()
    output, scenario = fn(*args)
    elapsed = time.time() - t0
    if scenario == "blocked":
        return f"⚠️ {output}"
    if scenario == "unknown":
        return output
    return f"{output}\n\n---\n({elapsed:.1f}s, {LLM_LABEL})"


def analyze_exam(user_input):
    return _timed(lambda u: run_pipeline(u), user_input)


def analyze_drug(user_input, uploaded_file, uploaded_file2):
    return _timed(lambda u, f1, f2: run_pipeline(u, f1, f2), user_input, uploaded_file, uploaded_file2)


def analyze_report(user_input, uploaded_file):
    msg = user_input.strip() or "请解读上传的体检报告中的异常指标"
    if uploaded_file is not None:
        return _timed(lambda u, f: run_pipeline(u, f), msg, uploaded_file)
    return _timed(lambda u: run_pipeline(u), msg)


def analyze_dept(user_input):
    return _timed(lambda u: run_pipeline(u), user_input)


# Build UI
with gr.Blocks(title="医疗健康智能助手") as app:
    gr.Markdown(f"# \U0001f3e5 医疗健康智能助手")
    gr.Markdown(f"**LLM**: {LLM_LABEL} | 仅供参考，不构成医疗建议")

    with gr.Tab("\U0001f50d 体检套餐选择"):
        gr.Markdown("基于健康画像推荐体检套餐。请先在 Streamlit 应用中保存画像。")
        exam_input = gr.Textbox(label="描述你的需求", placeholder="我35岁男程序员，预算1500，帮我推荐体检套餐", lines=2)
        exam_btn = gr.Button("开始分析", variant="primary")
        exam_output = gr.Markdown()
        exam_btn.click(fn=analyze_exam, inputs=exam_input, outputs=exam_output)

    with gr.Tab("\U0001f48a 药品对比"):
        gr.Markdown("输入药品名称或上传说明书图片，生成结构化对比。")
        drug_input = gr.Textbox(label="描述要对比的药品", placeholder="布洛芬和对乙酰氨基酚有什么区别？", lines=2)
        with gr.Row():
            drug_file1 = gr.File(label="药品A说明书（可选）", file_types=[".pdf", ".txt", ".jpg", ".jpeg", ".png"])
            drug_file2 = gr.File(label="药品B说明书（可选）", file_types=[".pdf", ".txt", ".jpg", ".jpeg", ".png"])
        drug_btn = gr.Button("开始对比", variant="primary")
        drug_output = gr.Markdown()
        drug_btn.click(fn=analyze_drug, inputs=[drug_input, drug_file1, drug_file2], outputs=drug_output)

    with gr.Tab("\U0001f4cb 体检报告解读"):
        gr.Markdown("描述异常指标或上传体检报告，获取解读建议。")
        rpt_input = gr.Textbox(label="描述异常指标", placeholder="谷丙转氨酶65偏高什么意思？挂什么科？", lines=2)
        rpt_file = gr.File(label="上传体检报告（可选，支持 PDF / TXT / 图片）", file_types=[".pdf", ".txt", ".jpg", ".jpeg", ".png"])
        rpt_btn = gr.Button("开始解读", variant="primary")
        rpt_output = gr.Markdown()
        rpt_btn.click(fn=analyze_report, inputs=[rpt_input, rpt_file], outputs=rpt_output)

    with gr.Tab("\U0001f3e5 就医科室推荐"):
        gr.Markdown("描述症状，推荐合适的就诊科室。")
        dept_input = gr.Textbox(label="描述你的症状", placeholder="头痛胸闷应该挂什么科？需要提前准备什么？", lines=2)
        dept_btn = gr.Button("开始推荐", variant="primary")
        dept_output = gr.Markdown()
        dept_btn.click(fn=analyze_dept, inputs=dept_input, outputs=dept_output)

    gr.Markdown("---\n⚠️ 免责声明：本项目为课程作业，所有分析仅供参考，不构成医疗建议。")

port = int(os.getenv("GRADIO_PORT", "7860"))
app.launch(server_name="0.0.0.0", server_port=port, show_error=True)
