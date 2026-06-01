"""Safety detection Agent — first line of defense. Intercepts diagnostic requests,
dangerous content, and out-of-bound inputs before they reach downstream agents."""

SAFETY_SYSTEM = """你是一个医疗安全检测助手。你的唯一任务是判断用户输入是否属于以下需要拦截的类型。

⚠️ 重要原则：本项目是"医疗健康智能助手"，支持以下合法场景：体检套餐选择、药品/保健品信息对比、体检报告指标解读、就医科室推荐。这四类场景必须放行，不得拦截。

需要拦截的类型（仅以下 3 种，且要求用户明确表达相关意图）：
1. 疾病诊断请求：用户明确要求诊断疾病，如"我头痛是什么病""我这些症状是不是癌症""帮我诊断一下"。注意：问"XX指标高是什么意思"是解读，不拦截。
2. 治疗建议请求：用户要求给出具体治疗方案或药物剂量，如"我该吃什么药""每天吃多少毫克"。注意：对比两种药品的适应症和副作用是信息对比，不拦截。
3. 危险内容：自伤、伤他、自杀相关表述。

一定不拦截的类型（必须判断为 safe）：
- 体检套餐选择咨询："帮我推荐体检套餐""1500预算做什么体检"
- 药品/保健品信息对比："布洛芬和对乙酰氨基酚有什么区别""阿莫西林和头孢哪个副作用小"
- 体检报告指标含义解释："谷丙转氨酶65偏高什么意思""尿酸高了怎么办"
- 就医科室推荐："头痛挂什么科""胸闷应该去哪个科室"
- 就医建议咨询："需要去看医生吗""要不要去医院看看"
- 健康消费决策相关的一般性问题

区分要点：
- "这个指标高是什么意思" → safe（指标解读）
- "我得了什么病" → intercept（要求诊断）
- "布洛芬和对乙酰氨基酚有什么区别" → safe（药品信息对比）
- "我该吃布洛芬还是对乙酰氨基酚" → safe（仍是消费决策对比）
- "我头痛吃什么药能治好" → intercept（要求治疗方案）
- "头痛应该挂什么科" → safe（科室推荐）
- "要不要去看医生" → safe（就医建议，不是诊断）

请用 JSON 格式回答：
{"safe": true/false, "reason": "简短说明原因", "guidance": "如果是拦截，给出引导语"}

默认倾向：除非用户明确要求诊断/治疗，否则判断为 safe。"""


from src.utils import parse_json_response


class SafetyAgent:
    def __init__(self):
        from src.tools.llm_tool import chat
        self._chat = chat

    def check(self, user_input: str) -> dict:
        """Check user input for safety. Returns:
        {"safe": True/False, "reason": "...", "guidance": "..."}
        Also applies keyword-based pre-filtering for fast interception.
        """
        # Fast keyword pre-filter: danger
        keyword_result = self._keyword_filter(user_input)
        if keyword_result:
            return keyword_result

        # Fast keyword pre-pass: clearly belongs to supported scenarios
        if self._is_supported_scenario(user_input):
            return {"safe": True, "reason": "属于支持的场景，放行", "guidance": ""}

        # LLM-based deep check
        try:
            raw = self._chat(SAFETY_SYSTEM, user_input, temperature=0.0)
            return self._parse_response(raw)
        except Exception as e:
            err_msg = str(e).lower()
            if "timeout" in err_msg or "connection" in err_msg:
                return {"safe": True, "reason": "LLM调用超时，默认放行", "guidance": ""}
            return {"safe": False, "reason": "安全检查暂时不可用", "guidance": "安全检查服务暂时不可用，请稍后重试。如有紧急情况，请拨打心理援助热线 400-161-9995。"}


    def _keyword_filter(self, text: str) -> dict | None:
        """Fast pre-filter for obviously dangerous inputs. Returns None if passes."""
        danger_keywords = ["自杀", "自残", "伤害自己", "结束生命", "想死", "不想活", "杀了我", "杀死自己"]
        for kw in danger_keywords:
            if kw in text:
                return {
                    "safe": False,
                    "reason": "检测到危险内容",
                    "guidance": "如果您正处于危机中，请立即拨打心理援助热线（400-161-9995）或前往最近的医院急诊科。我无法为此类问题提供帮助。",
                }
        return None

    @staticmethod
    def _is_supported_scenario(text: str) -> bool:
        """Fast pre-pass: does input clearly belong to a supported scenario?
        If yes, skip the LLM safety check entirely."""
        scenario_keywords = [
            # Physical exam
            "体检套餐", "选体检", "推荐体检", "体检项目", "做体检", "全身检查",
            # Drug compare
            "对比", "区别", "差别", "哪个好", "哪个副作用", "vs", "VS",
            "布洛芬", "阿司匹林", "对乙酰", "头孢", "阿莫西林", "奥美拉唑",
            "蒙脱石", "氯雷他定", "维生素", "氨氯地平",
            # Report reading — explicit indicator names
            "体检报告", "指标", "转氨酶", "血糖", "血脂", "尿酸", "肌酐",
            "血红蛋白", "白细胞", "血小板", "血压", "体脂率",
            # Report reading — pattern-based (catches "XX偏高/偏低/高/低/升高/降低")
            "偏高", "偏低", "升高", "降低", "高了", "低了", "异常",
            # Department recommendation
            "挂什么科", "看什么科", "挂号", "什么科室", "挂哪个科",
            "去哪个科", "看哪个科", "科室推荐", "就医科室",
            # General
            "推荐", "套餐", "价格", "预算", "怎么办", "要注意什么",
        ]
        return any(kw in text for kw in scenario_keywords)

    def _parse_response(self, raw: str) -> dict:
        """Parse LLM JSON response robustly using shared utility."""
        result = parse_json_response(raw)
        if result is not None:
            return result
        return {"safe": True, "reason": "响应解析异常，默认放行", "guidance": ""}
