"""Reflection & verification Agent — 5-dimension output quality check."""

REFLECTION_SYSTEM = """你是一个医疗内容质量审核助手。你的任务是对 AI 生成的医疗消费建议进行五维检查。

检查维度：
1. **边界检查**：输出中是否含有明确的诊断结论或治疗建议？
   - 不合格的例子："你得了脂肪肝""你这是冠心病""你应该吃阿莫西林""建议服用XX药物治疗""属于确诊""确诊为""诊断为"
   - 合格的例子（以下不算越界）："该指标升高常见与肝脏功能相关""偏离参考范围较明显，建议就医进一步评估""建议咨询消化内科""建议定期复查""需要医生进一步明确"
   - 判断标准：描述指标与器官/系统的关联 = 合格；断言用户患有具体疾病 = 不合格；建议就医/复查 = 合格（这是健康指引不是治疗建议）；建议服用具体药物 = 不合格；出现"确诊""诊断"等疾病确认类词汇 = 不合格
2. **一致性检查**：输出的事实性信息是否和提供的知识库内容一致？有无编造的数据？
3. **安全性检查**：是否含有诱导性词汇？
   - 不合格的词汇："首选""最好""一定""保证""绝对安全""必然会""肯定有效"
   - 合格的词汇（以下不算违规）："推荐""建议""值得关注""可以考虑""常见""一般""通常"
4. **免责检查**：是否以"仅供参考"类声明结尾？
5. **完整性检查**：对比表/解读是否遗漏了关键维度（如药品对比缺少禁忌项）？

请用 JSON 格式回答：
{
  "pass": true/false,
  "checks": {
    "boundary": {"pass": true/false, "issue": "具体问题或空"},
    "consistency": {"pass": true/false, "issue": "具体问题或空"},
    "safety": {"pass": true/false, "issue": "具体问题或空"},
    "disclaimer": {"pass": true/false, "issue": "具体问题或空"},
    "completeness": {"pass": true/false, "issue": "具体问题或空"}
  },
  "summary": "通过所有检查" 或 "需要修正的问题汇总"
}"""


from src.utils import parse_json_response


class ReflectionAgent:
    def __init__(self, max_retries: int = 2):
        from src.tools.llm_tool import chat
        self._chat = chat
        self.max_retries = max_retries

    def verify(self, output: str, retrieved_context: str = "") -> dict:
        """
        Verify output quality. Returns:
            {"pass": bool, "checks": dict, "summary": str}
        """
        prompt = f"知识库检索内容：\n{retrieved_context or '(无)'}\n\n待审核的 AI 输出：\n{output}"

        try:
            raw = self._chat(REFLECTION_SYSTEM, prompt, temperature=0.0)
            return self._parse_response(raw)
        except Exception:
            return {"pass": False, "checks": {"error": {"pass": False, "issue": "校验Agent异常"}}, "summary": "校验Agent调用失败，需人工检查"}

    def _parse_response(self, raw: str) -> dict:
        result = parse_json_response(raw)
        if result is not None:
            return result
        return {"pass": False, "checks": {"parse": {"pass": False, "issue": "校验响应解析失败"}}, "summary": "校验Agent响应格式异常，需人工检查"}
