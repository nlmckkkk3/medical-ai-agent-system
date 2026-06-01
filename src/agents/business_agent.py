"""Scenario business Agent — orchestrates 4 core scenarios:
physical exam selection, drug comparison, report reading, and department recommendation.

Each scenario follows: info completion → RAG retrieval → analysis → output."""

from pathlib import Path

from src.tools.llm_tool import chat, chat_stream
from src.tools.rag_tool import RAGTool
from src.tools.doc_parser import DocParser
from src.memory.user_memory import UserMemory, UserProfile


PHYSICAL_EXAM_SYSTEM = """你是体检套餐推荐顾问。基于用户画像和知识库信息推荐最合适的套餐。

核心规则（必须严格遵守）：
1. 只使用知识库中已有的套餐信息，绝不编造套餐名称、价格、检查项目数或适用人群
2. 如果知识库中所有套餐都不适合该用户（如年龄不匹配、预算差距过大），必须如实告知用户"当前知识库中没有完全匹配您条件的套餐"，然后列出最接近的选项并说明不匹配的原因
3. 不要为了完成推荐任务而强行推荐不匹配的套餐
4. 信息缺失时标注"暂无数据"，不要猜测或编造
5. 不说"你应该做XX检查"；定位为健康决策辅助"""

DRUG_COMPARE_SYSTEM = """你是药品信息对比助手。将药品信息整理成专业Markdown表格对比。

信息来源优先级：
1. 知识库检索结果（经过验证的可靠数据）→ 前提：知识库中包含用户查询的药品
2. 用户上传的文件内容（药品说明书）→ 作为补充
3. 你的药学训练知识 → 知识库和文件中都没有用户查询的药品时使用

核心规则（必须严格遵守）：
1. 最重要的规则：只分析用户明确提到的药品，绝不输出用户没有询问的药品
2. 知识库中有用户查询的药品 → 优先使用知识库数据，不要用训练知识替代
3. 知识库中没有用户查询的药品 → 忽略知识库内容（即使有其他药品数据），用你的药学知识进行对比，并在表格前明确标注"⚠️ 以下药品信息来自AI知识库，可能与实际药品说明书存在差异，请以药品包装内说明书为准"
4. 如果无法确定用户想查询的药品名称 → 请用户明确说明药品名称，不要猜测或展示无关药品
5. 如果用户只提到一种药品：展示该药品信息并询问是否需要对比其他药品；可推荐1-2个同类常见药品供用户选择是否对比
6. 信息不确定处标注"暂无数据"或"待核实"，不要编造具体数字

输出格式要求：
1. 使用 | 维度 | 药品A | 药品B | 的对比表格，维度列加粗
2. 表格后附"对比总结"段落，用要点列表总结关键差异
3. 信息缺失处标注"暂无数据"而非留空

规则：标注"仅供参考"；不推荐哪个更优，只呈现事实差异；不说"你应该吃XX药"。"""

REPORT_READING_SYSTEM = """你是体检报告解读助手。解释异常指标含义并建议就诊科室。

核心规则：
1. 绝不说"你得了XX病"；绝不在输出中列出具体疾病名称作为可能原因（如"脂肪肝""肝炎""冠心病"）；只描述"该指标升高可能与肝脏功能相关"，建议用户咨询对应科室由医生判断
2. 必须注明仅供参考
3. OCR 识别上传的报告时数值可能有误差，引用具体数值时要谨慎。如果某个数值与临床常识严重不符（如超出参考范围数倍），应标注"⚠️ 该数值可能因图片识别有误，建议核对原始报告"
4. 不要编造具体的医学干预阈值（如"超过XXX需要进一步检查"），除非用户明确提供了该参考范围
5. 对每个异常指标进行偏离程度评估（内部计算，不在输出中展示数值计算过程）：偏离幅度 = |实测值 - 最近的参考限| / 参考限。按以下原则给出就医建议：
   - 偏离 <20%：表述为"轻微偏离参考范围，建议定期复查观察变化"
   - 偏离 20-50%：表述为"该指标偏离参考范围较明显，值得关注，建议安排就医进一步评估"
   - 偏离 >50% 或达到危急值：表述为"该指标偏离参考范围较多，建议尽快就医由医生全面评估"
   措辞规范：只使用上述推荐表述，绝不说"轻度异常/中度异常/重度异常/确诊"。目的是让用户了解严重程度但不引起不必要的恐慌
6. 解读贫血相关指标时，只描述指标间的关联模式（如"小细胞低色素性"），不写具体疾病推断
7. 绝不在输出中使用"确诊""诊断""确认患病""肯定是""必然是"等词汇，即使是在否定语境中也要避免
8. 避免使用"首选""最好""一定""保证"等诱导性词汇
9. 输出格式：先逐项解读异常指标，再给出综合建议和就诊科室推荐"""

DEPARTMENT_SYSTEM = """你是就医科室推荐助手。根据用户描述的症状，推荐合适的就诊科室。
输出格式：先简要分析症状可能涉及的系统（只说系统名称如"心血管系统""消化系统"，不列具体疾病名），然后推荐1-3个优先就诊科室（只说明每个科室的诊治范围，不说"可能是XX病"），最后给出就诊前准备建议。
严格规则：
1. 不诊断疾病：绝不说"可能是XX病""可能是XX症""可能是XX的信号""提示XX问题"
2. 正确表述示范："头痛是神经内科的常见就诊症状，医生会通过检查评估您的情况"；错误表述："头痛可能是偏头痛或高血压的信号"
3. 避免使用"首选""最好""一定""保证"等诱导性词汇
4. 必须注明仅供参考"""

DISCLAIMER = "\n\n---\n*以上信息基于公开资料生成，仅供参考，请结合医生/药师建议做最终决定。*"

DISCLAIMER_AI_KNOWLEDGE = "\n\n---\n⚠️ *以上药品信息部分来自AI知识库，未经人工逐一验证，可能与实际药品说明书存在差异。请务必以药品包装内说明书为准，用药前咨询医生或药师。*"


class BusinessAgent:
    def __init__(self):
        self.rag = RAGTool()
        self.doc_parser = DocParser()
        self.memory = UserMemory()

    @staticmethod
    def _truncate(text: str, max_chars: int = 2000) -> str:
        return text if len(text) <= max_chars else text[:max_chars] + "..."

    @staticmethod
    def _build_history_context(history: list[dict] = None, scenario: str = "") -> str:
        """Build context string from recent conversation history, filtered by scenario."""
        if not history:
            return ""
        # Filter: only include entries matching current scenario (or untagged legacy entries)
        filtered = [
            h for h in history
            if h.get("scenario", scenario) == scenario
        ]
        if not filtered:
            return ""
        recent = filtered[-4:]  # Last 4 exchanges within this scenario
        lines = ["最近的对话："]
        for h in recent:
            role = "用户" if h["role"] == "user" else "助手"
            content = h["content"][:200]
            lines.append(f"{role}: {content}")
        return "\n".join(lines) + "\n\n当前问题："

    def run(self, scenario: str, user_input: str, user_id: str = "default",
            uploaded_file: str = None, history: list[dict] = None,
            uploaded_file2: str = None) -> dict:
        """
        Execute the appropriate scenario workflow. Returns:
            {"scenario": str, "output": str, "need_profile": bool, "missing_fields": list}
        """
        profile = self.memory.load(user_id)
        history_context = self._build_history_context(history, scenario)

        if scenario == "physical_exam":
            return self._do_physical_exam(user_input, profile, history_context)
        elif scenario == "drug_compare":
            return self._do_drug_compare(user_input, uploaded_file, history_context, uploaded_file2)
        elif scenario == "report_reading":
            return self._do_report_reading(user_input, uploaded_file, history_context, profile)
        elif scenario == "department_recommendation":
            return self._do_department_recommendation(user_input, history_context, profile)
        else:
            return {"scenario": "unknown", "output": "", "need_profile": False, "missing_fields": []}

    def run_stream(self, scenario: str, user_input: str, user_id: str = "default",
                   uploaded_file: str = None, history: list[dict] = None,
                   uploaded_file2: str = None):
        """Stream version of run(). Yields content chunks for st.write_stream."""
        profile = self.memory.load(user_id)
        history_context = self._build_history_context(history, scenario)

        if scenario == "physical_exam":
            if not profile.is_complete():
                missing = profile.missing_fields()
                disclaimer = f"为了给您更精准的推荐，我还需要了解以下信息：{'、'.join(missing)}。请补充后继续。"
                yield disclaimer
                return

            profile_str = (
                f"年龄：{profile.age}岁\n性别：{profile.gender}\n"
                f"职业：{profile.occupation}\n预算：{profile.budget}元\n"
                f"家族病史：{profile.family_history or '未提供'}\n"
                f"已有慢性病：{profile.chronic_conditions or '未提供'}"
            )
            # Use raw user input for RAG search (may override profile, e.g. checking for family)
            context = self._truncate(self.rag.search_formatted("physical_exams", user_input, top_k=8))
            if not context:
                yield f"当前知识库中暂未找到匹配您条件的体检套餐。\n\n您的当前画像：\n{profile_str}\n\n建议：1) 尝试更具体的描述（如'中青年男性体检套餐'） 2) 调整预算范围"
                return
            prompt = (
                f"用户画像（仅供参考，如与用户需求冲突以需求为准）：\n{profile_str}\n\n"
                + history_context
                + f"用户需求：{user_input}\n\n"
                f"知识库检索结果（这是知识库中仅有的套餐信息，请勿编造其他套餐）：\n{context}\n\n"
                "重要规则：以用户需求中的年龄/性别/预算为准（用户可能在帮家人朋友查询），画像仅作背景参考。\n"
                "请基于以上真实套餐信息进行推荐。如果知识库中没有适合该用户的套餐（如年龄不匹配、预算差距大），"
                "请如实告知用户，列出最接近的选项并说明不匹配原因，绝不编造套餐。"
            )
            yield from chat_stream(PHYSICAL_EXAM_SYSTEM, prompt)
            yield DISCLAIMER

        elif scenario == "drug_compare":
            file_text_a = ""
            file_text_b = ""
            file_name_a = ""
            file_name_b = ""
            any_ocr_failed = False

            if uploaded_file:
                file_text_a = self.doc_parser.parse_or_empty(uploaded_file)
                if not file_text_a:
                    any_ocr_failed = True
                file_name_a = Path(uploaded_file).stem

            if uploaded_file2:
                file_text_b = self.doc_parser.parse_or_empty(uploaded_file2)
                if not file_text_b:
                    any_ocr_failed = True
                file_name_b = Path(uploaded_file2).stem

            context = self._truncate(self.rag.search_formatted("drugs", user_input, top_k=5))

            if any_ocr_failed:
                source_note = (
                    "请用你的药学知识解读用户提到的药品。"
                    "对于成分、适应症、不良反应、注意事项等事实信息，可以基于你的药学知识自信地输出，不要全部填'暂无数据'。"
                    "只有价格、具体用法用量等无法确定的内容才标注'暂无数据'。"
                    "在输出开头注明信息来源。"
                )
                disclaimer = DISCLAIMER_AI_KNOWLEDGE
            elif context or file_text_a or file_text_b:
                source_note = "请使用知识库和文件中的药品信息。只分析用户查询的药品，忽略知识库中不相关的药品数据。"
                disclaimer = DISCLAIMER
            else:
                source_note = (
                    "请用你的药学知识解读用户提到的药品。"
                    "对于成分、适应症、不良反应、注意事项等事实信息，可以基于你的药学知识自信地输出，不要全部填'暂无数据'。"
                    "只有价格、具体用法用量等无法确定的内容才标注'暂无数据'。"
                    "在输出开头注明信息来源。"
                )
                disclaimer = DISCLAIMER_AI_KNOWLEDGE

            kb_section = f"知识库检索结果（注意：只使用与用户查询匹配的药品，不相关的内容必须忽略）：\n{context}\n\n" if context else "（知识库中暂无相关药品数据）\n\n"
            file_parts = []
            if file_text_a:
                file_parts.append(f"【药品A说明书：{file_name_a}】\n{file_text_a}")
            if file_text_b:
                file_parts.append(f"【药品B说明书：{file_name_b}】\n{file_text_b}")
            file_section = "\n\n".join(file_parts) + "\n\n" if file_parts else ""
            prompt = (
                history_context
                + f"用户需求：{user_input}\n\n"
                + file_section
                + kb_section
                + source_note
            )
            yield from chat_stream(DRUG_COMPARE_SYSTEM, prompt)
            yield disclaimer

        elif scenario == "report_reading":
            file_text = ""
            if uploaded_file:
                file_text = self.doc_parser.parse_or_empty(uploaded_file)

            # OCR failed: give user clear guidance instead of passing empty content to LLM
            if uploaded_file and not file_text:
                yield (
                    "未能从上传的文件中自动识别出文字内容。\n\n"
                    "您可以尝试以下替代方式：\n"
                    "① 直接在输入框中输入异常指标名称和数值（例如：谷丙转氨酶 65，参考范围 0-40）\n"
                    "② 将报告保存为 PDF 格式后重新上传（识别效果更好）\n"
                    "③ 重新拍照，确保光线充足、文字清晰、正对纸张\n\n"
                    "我会根据您提供的信息帮您解读指标含义并提供就医建议。"
                )
                return

            search_query = user_input if user_input else (file_text[:500] if file_text else "")
            context = self._truncate(self.rag.search_formatted("medical_knowledge", search_query, top_k=3))
            profile_context = (
                f"用户画像：{profile.age}岁，{profile.gender}，"
                f"{'慢性病: ' + profile.chronic_conditions + '。' if profile.chronic_conditions else '无已知慢性病。'}"
                f"{'家族病史: ' + profile.family_history + '。' if profile.family_history else ''}\n"
            )
            prompt = (
                history_context
                + profile_context
                + f"用户问题：{user_input}\n\n"
                + (f"报告内容：\n{file_text}\n\n" if file_text else "")
                + f"知识库参考：\n{context}\n\n"
                + "请解读异常指标，解释可能含义，建议就诊科室。注意不要做诊断性结论。"
            )
            yield from chat_stream(REPORT_READING_SYSTEM, prompt)
            yield DISCLAIMER

        elif scenario == "department_recommendation":
            context = self._truncate(self.rag.search_formatted("departments", user_input, top_k=3))
            profile_context = (
                f"用户画像：{profile.age}岁，{profile.gender}，"
                f"{'慢性病: ' + profile.chronic_conditions + '。' if profile.chronic_conditions else '无已知慢性病。'}"
                f"{'家族病史: ' + profile.family_history + '。' if profile.family_history else ''}\n"
            )
            prompt = (
                history_context
                + profile_context
                + f"用户描述的症状：{user_input}\n\n"
                f"知识库参考：\n{context}\n\n"
                "请根据症状和用户画像推荐合适的就诊科室，说明每个科室的诊治范围和建议理由。"
            )
            yield from chat_stream(DEPARTMENT_SYSTEM, prompt)
            yield DISCLAIMER

    def _do_physical_exam(self, user_input: str, profile: UserProfile,
                          history_context: str = "") -> dict:
        # Step 1: Check profile completeness
        if not profile.is_complete():
            missing = profile.missing_fields()
            return {
                "scenario": "physical_exam",
                "output": f"为了给您更精准的推荐，我还需要了解以下信息：{'、'.join(missing)}。请补充后继续。",
                "need_profile": True,
                "missing_fields": missing,
            }

        # Step 2: Build profile string
        profile_str = (
            f"年龄：{profile.age}岁\n"
            f"性别：{profile.gender}\n"
            f"职业：{profile.occupation}\n"
            f"预算：{profile.budget}元\n"
            f"家族病史：{profile.family_history or '未提供'}\n"
            f"已有慢性病：{profile.chronic_conditions or '未提供'}"
        )

        # Step 3: RAG search — use raw user input (may override profile, e.g. checking for family)
        context = self._truncate(self.rag.search_formatted("physical_exams", user_input, top_k=8))
        if not context:
            return {
                "scenario": "physical_exam",
                "output": (
                    "当前知识库中暂未找到匹配您条件的体检套餐。\n\n"
                    "建议：1) 放宽预算范围 2) 告诉我您更关注哪些检查项目（如心脑血管、肿瘤筛查等）\n\n"
                    f"您的当前画像：\n{profile_str}"
                ),
                "need_profile": False,
                "missing_fields": [],
            }

        # Step 4: Generate recommendation
        prompt = (
            f"用户画像（仅供参考，如与用户需求冲突以需求为准）：\n{profile_str}\n\n"
            + history_context
            + f"用户需求：{user_input}\n\n"
            f"知识库检索结果（这是知识库中仅有的套餐信息，请勿编造其他套餐）：\n{context}\n\n"
            "重要规则：以用户需求中的年龄/性别/预算为准（用户可能在帮家人朋友查询），画像仅作背景参考。\n"
            "请基于以上真实套餐信息进行推荐。如果知识库中没有适合该用户的套餐（如年龄不匹配、预算差距大），"
            "请如实告知用户，列出最接近的选项并说明不匹配原因，绝不编造套餐。"
        )
        output = chat(PHYSICAL_EXAM_SYSTEM, prompt)
        output += DISCLAIMER

        return {
            "scenario": "physical_exam",
            "output": output,
            "need_profile": False,
            "missing_fields": [],
        }

    def _do_drug_compare(self, user_input: str, uploaded_file: str = None,
                         history_context: str = "", uploaded_file2: str = None) -> dict:
        file_text_a = ""
        file_text_b = ""
        file_name_a = ""
        file_name_b = ""
        any_ocr_failed = False

        if uploaded_file:
            file_text_a = self.doc_parser.parse_or_empty(uploaded_file)
            if not file_text_a:
                any_ocr_failed = True
            file_name_a = Path(uploaded_file).stem

        if uploaded_file2:
            file_text_b = self.doc_parser.parse_or_empty(uploaded_file2)
            if not file_text_b:
                any_ocr_failed = True
            file_name_b = Path(uploaded_file2).stem

        # RAG search for drug info
        context = self._truncate(self.rag.search_formatted("drugs", user_input, top_k=5))

        # Build prompt based on available data sources
        if any_ocr_failed:
            source_note = (
                "请用你的药学知识解读用户提到的药品。"
                "对于成分、适应症、不良反应、注意事项等事实信息，可以基于你的药学知识自信地输出，不要全部填'暂无数据'。"
                "只有价格、具体用法用量等无法确定的内容才标注'暂无数据'。"
                "在输出开头注明信息来源。"
            )
            disclaimer = DISCLAIMER_AI_KNOWLEDGE
        elif context or file_text_a or file_text_b:
            source_note = "请使用知识库和文件中的药品信息。只分析用户查询的药品，忽略知识库中不相关的药品数据。"
            disclaimer = DISCLAIMER
        else:
            source_note = (
                "请用你的药学知识解读用户提到的药品。"
                "对于成分、适应症、不良反应、注意事项等事实信息，可以基于你的药学知识自信地输出，不要全部填'暂无数据'。"
                "只有价格、具体用法用量等无法确定的内容才标注'暂无数据'。"
                "在输出开头注明信息来源。"
            )
            disclaimer = DISCLAIMER_AI_KNOWLEDGE

        kb_section = f"知识库检索结果（注意：只使用与用户查询匹配的药品，不相关的内容必须忽略）：\n{context}\n\n" if context else "（知识库中暂无相关药品数据）\n\n"
        file_parts = []
        if file_text_a:
            file_parts.append(f"【药品A说明书：{file_name_a}】\n{file_text_a}")
        if file_text_b:
            file_parts.append(f"【药品B说明书：{file_name_b}】\n{file_text_b}")
        file_section = "\n\n".join(file_parts) + "\n\n" if file_parts else ""
        prompt = (
            history_context
            + f"用户需求：{user_input}\n\n"
            + file_section
            + kb_section
            + source_note
        )
        output = chat(DRUG_COMPARE_SYSTEM, prompt)
        output += disclaimer

        return {
            "scenario": "drug_compare",
            "output": output,
            "need_profile": False,
            "missing_fields": [],
        }

    def _do_report_reading(self, user_input: str, uploaded_file: str = None,
                           history_context: str = "", profile: UserProfile = None) -> dict:
        file_text = ""
        if uploaded_file:
            file_text = self.doc_parser.parse_or_empty(uploaded_file)
            if not file_text:
                return {
                    "scenario": "report_reading",
                    "output": (
                        "未能从上传的文件中自动识别出文字内容。\n\n"
                        "您可以尝试以下替代方式：\n"
                        "① 直接在输入框中输入异常指标名称和数值（例如：谷丙转氨酶 65，参考范围 0-40）\n"
                        "② 将报告保存为 PDF 格式后重新上传（识别效果更好）\n"
                        "③ 重新拍照，确保光线充足、文字清晰、正对纸张\n\n"
                        "我会根据您提供的信息帮您解读指标含义并提供就医建议。"
                    ),
                    "need_profile": False,
                    "missing_fields": [],
                }

        # RAG search for indicator explanations
        search_query = user_input
        if not search_query and file_text:
            search_query = file_text[:500]
        context = self._truncate(self.rag.search_formatted("medical_knowledge", search_query, top_k=3))

        profile_context = ""
        if profile:
            profile_context = (
                f"用户画像：{profile.age}岁，{profile.gender}，"
                f"{'慢性病: ' + profile.chronic_conditions + '。' if profile.chronic_conditions else '无已知慢性病。'}"
                f"{'家族病史: ' + profile.family_history + '。' if profile.family_history else ''}\n"
            )
        prompt = (
            history_context
            + profile_context
            + f"用户问题：{user_input}\n\n"
            + (f"报告内容：\n{file_text}\n\n" if file_text else "")
            + f"知识库参考：\n{context}\n\n"
            + "请结合用户画像解读异常指标，解释可能含义，建议就诊科室。注意不要做诊断性结论。"
        )
        output = chat(REPORT_READING_SYSTEM, prompt)
        output += DISCLAIMER

        return {
            "scenario": "report_reading",
            "output": output,
            "need_profile": False,
            "missing_fields": [],
        }

    def _do_department_recommendation(self, user_input: str,
                                      history_context: str = "",
                                      profile: UserProfile = None) -> dict:
        context = self._truncate(self.rag.search_formatted("departments", user_input, top_k=3))
        profile_context = ""
        if profile:
            profile_context = (
                f"用户画像：{profile.age}岁，{profile.gender}，"
                f"{'慢性病: ' + profile.chronic_conditions + '。' if profile.chronic_conditions else '无已知慢性病。'}"
                f"{'家族病史: ' + profile.family_history + '。' if profile.family_history else ''}\n"
            )
        prompt = (
            history_context
            + profile_context
            + f"用户描述的症状：{user_input}\n\n"
            f"知识库参考：\n{context}\n\n"
            "请结合用户画像和症状推荐合适的就诊科室，说明每个科室的诊治范围和建议理由。"
        )
        output = chat(DEPARTMENT_SYSTEM, prompt)
        output += DISCLAIMER
        return {
            "scenario": "department_recommendation",
            "output": output,
            "need_profile": False,
            "missing_fields": [],
        }

    def update_profile(self, user_id: str, **kwargs) -> UserProfile:
        """Update user profile fields and save."""
        profile = self.memory.load(user_id)
        for key, value in kwargs.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
        profile.user_id = user_id
        self.memory.save(profile)
        return profile

    def get_profile(self, user_id: str = "default") -> UserProfile:
        return self.memory.load(user_id)
