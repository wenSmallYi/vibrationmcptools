import json
from functions.llm_engine_endpoint import OpenAIEngine

class AIAgent:
    def __init__(self):
        self.engine = OpenAIEngine()

    def parse_instruction(self, user_input: str) -> dict:
        prompt = f'''
        你是一個振動訊號分析指令助手，請根據使用者輸入的自然語言，解析為 JSON 指令格式：

        {{
        "axis": "X" 或 "Y" 或 "Z",
        "processing": {{
            "method": "bandpass_filter",
            "params": {{ "low": 頻率下限, "high": 頻率上限 }}
        }},
            "features": ["RMS", "Skewness", "Kurtosis", "CrestFactor", "Estimated Speed"]
        }}

        請根據使用者的描述進行合理判斷。未提到的欄位可不給出，輸入如下：
        {user_input}
        '''
        messages = [{"role": "user", "content": prompt}]
        reply = self.engine.generate(messages)

        try:
            json_str = reply.strip().split("```")[0] if "```" in reply else reply
            return json.loads(json_str)
        except Exception:
            return {}

    def generate_summary(self, features: dict) -> str:
        prompt = (
            "以下是機械系統振動訊號分析的統計特徵資料，請用專業工程/數據分析語言進行評估與建議：\n\n"
            f"{features}\n\n"
            "請說明機械系統是否正常、有無異常原因、是否需要維修，以及可能的建議。"
        )
        messages = [{"role": "user", "content": prompt}]
        return self.engine.generate(messages)
