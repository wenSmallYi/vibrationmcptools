import json
from vibration_mcp_llm.llm_engine_endpoint_v2 import OpenAIEngine

class AIAgent:
    def __init__(self):
        self.engine = OpenAIEngine()

    def generate_summary(self, all_results: list) -> str:
        prompt = (
            "以下是振動訊號多組統計/頻譜特徵資料（每組可能有不同軸、不同濾波條件），請用專業工程/數據分析語言，綜合評估每組結果、判斷異常與維修建議，最後做總結：\n\n"
            f"{json.dumps(all_results, ensure_ascii=False, indent=2)}\n\n"
            "請說明各組條件下系統狀態、異常原因與建議。"
        )
        messages = [{"role": "user", "content": prompt}]
        return self.engine.generate(messages)
