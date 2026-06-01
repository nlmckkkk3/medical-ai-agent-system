"""Intent routing Agent — classifies user intent and routes to the correct scenario."""

ROUTER_SYSTEM = """你是一个医疗健康智能助手的意图分类模块。你的任务是将用户输入分类到以下场景之一。

场景类别：
- "physical_exam"：体检套餐选择。用户想选体检套餐、对比体检项目、咨询体检相关事宜
- "drug_compare"：药品/保健品对比。用户想对比两种或多种药品/保健品
- "report_reading"：体检报告解读。用户描述了体检指标/身体数据，想了解含义
- "department_recommendation"：就医科室推荐。用户描述症状问应该挂什么科、看什么科室

⚠️ 重要规则：
- 不要轻易判定为 unknown。只要用户输入涉及医疗健康话题，就选最接近的场景
- 描述症状+问"怎么办" → report_reading
- 提到药品名称 → drug_compare
- 提到体检/套餐/预算 → physical_exam
- 提到挂号/科室/去哪个科 → department_recommendation
- 只有完全无关的输入（如"今天天气不错""帮我写代码"）才判为 unknown

请用 JSON 格式回答：
{"scenario": "分类标签", "confidence": 0.0-1.0}"""


from src.utils import parse_json_response


class RouterAgent:
    def __init__(self):
        from src.tools.llm_tool import chat
        self._chat = chat

    def route(self, user_input: str) -> dict:
        """Classify user intent. Returns:
        {"scenario": str, "confidence": float}"""
        keyword_result = self._keyword_route(user_input)
        if keyword_result:
            return keyword_result

        # Secondary: general medical keywords → likely report_reading
        if self._is_medical_query(user_input):
            return {"scenario": "report_reading", "confidence": 0.6}

        try:
            raw = self._chat(ROUTER_SYSTEM, user_input, temperature=0.0)
            result = self._parse_response(raw)
            # If LLM says unknown but query looks medical-ish, default to report_reading
            if result["scenario"] == "unknown" and self._is_medical_query(user_input):
                return {"scenario": "report_reading", "confidence": 0.4}
            return result
        except Exception:
            if self._is_medical_query(user_input):
                return {"scenario": "report_reading", "confidence": 0.3}
            return {"scenario": "unknown", "confidence": 0.0}

    def _keyword_route(self, text: str) -> dict | None:
        """Fast keyword-based routing for obvious cases."""
        # Each keyword has a weight — scenario-specific terms weight higher
        exam_keywords = {
            "体检套餐": 3, "选体检": 3, "推荐体检": 3, "体检项目": 3, "做体检": 3,
            "全身检查": 2,
        }
        drug_keywords = {
            "对比": 2, "vs": 3, "VS": 3, "哪个副作用": 3, "说明书": 2,
            "布洛芬": 3, "阿司匹林": 3, "对乙酰": 3, "头孢": 3, "阿莫西林": 3,
            "奥美拉唑": 3, "蒙脱石": 3, "氯雷他定": 3, "氨氯地平": 3,
        }
        report_keywords = {
            "体检报告": 3, "指标": 2, "报告单": 3,
            "偏高": 3, "偏低": 3, "升高": 3, "降低": 3, "高了": 3, "低了": 3,
            "异常": 2, "正常范围": 3, "正常值": 3,
            "转氨酶": 3, "血糖": 3, "血脂": 3, "尿酸": 3, "肌酐": 3,
            "血红蛋白": 3, "白细胞": 3, "血小板": 3, "血压": 3, "体脂率": 3,
            "胆固醇": 3, "甘油三酯": 3,
        }
        dept_keywords = {
            "挂什么科": 3, "看什么科": 3, "挂号": 2, "什么科室": 3, "挂哪个科": 3,
            "去哪个科": 3, "看哪个科": 3, "科室推荐": 3, "就医科室": 3,
            "看什么医生": 3, "挂什么号": 3, "哪个科室": 3, "就诊": 2,
        }

        scores = {
            "physical_exam": sum(w for kw, w in exam_keywords.items() if kw in text),
            "drug_compare": sum(w for kw, w in drug_keywords.items() if kw in text),
            "report_reading": sum(w for kw, w in report_keywords.items() if kw in text),
            "department_recommendation": sum(w for kw, w in dept_keywords.items() if kw in text),
        }

        best_scenario = max(scores, key=scores.get)
        best_score = scores[best_scenario]

        if best_score == 0:
            return None
        return {"scenario": best_scenario, "confidence": min(0.9, 0.5 + best_score * 0.1)}

    @staticmethod
    def _is_medical_query(text: str) -> bool:
        """Check if input contains any medical/health related terms.
        Used as fallback to avoid 'unknown' for medical queries."""
        medical_terms = [
            # Body parts & symptoms
            "头痛", "头晕", "胸闷", "胸痛", "腹痛", "腰痛", "背痛", "关节",
            "咳嗽", "发烧", "发热", "恶心", "呕吐", "腹泻", "便秘", "失眠",
            "乏力", "疲劳", "肿胀", "疼痛", "麻木", "痒", "皮疹", "脱发",
            # Medical indicators / concepts
            "体脂", "体重", "身高", "BMI", "心率", "脉搏", "体温", "血", "尿",
            "肝胆", "肾", "胃", "肠", "肺", "心脏", "肝", "脾", "胰",
            # Health-related
            "怎么办", "要注意什么", "怎么改善", "怎么调理", "怎么降",
            "什么是", "是什么意思", "是什么原因", "严不严重", "要不要紧",
            "正常吗", "需要", "应该", "建议",
            # Drugs / supplements
            "药", "保健品", "钙片", "鱼油",
            # Medical departments
            "内科", "外科", "科",
        ]
        return any(term in text for term in medical_terms)

    def _parse_response(self, raw: str) -> dict:
        result = parse_json_response(raw)
        if result is not None:
            return result
        return {"scenario": "unknown", "confidence": 0.0}

    @staticmethod
    def get_guidance(scenario: str) -> str:
        """Return user-facing guidance when intent is unclear."""
        return (
            "我没能确定您具体需要什么帮助。您可以这样问我：\n\n"
            "1. **体检套餐选择**：如「我30岁女性，预算1000，推荐什么体检套餐？」\n"
            "2. **药品对比**：如「请对比布洛芬和对乙酰氨基酚的区别」\n"
            "3. **体检报告解读**：如「我的体检报告显示尿酸偏高，是什么意思？」\n"
            "4. **就医科室推荐**：如「头痛胸闷应该挂什么科？」"
        )
