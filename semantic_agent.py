import json
from functions.llm_engine_endpoint import OpenAIEngine

class SemanticAgent:
    def __init__(self):
        self.engine = OpenAIEngine()

    def extract_semantics(self, user_input: str) -> dict:
        prompt = f"""
請從使用者輸入的文字中提取出振動訊號分析的參數，請回傳以下 JSON 格式：

{{
  "axis": "X"、"Y"、或 "Z"（若未提及則略過）,
  "low_freq": 帶通濾波器下限（若有）,
  "high_freq": 帶通濾波器上限（若有）,
  "features": ["RMS", "Kurtosis", "Skewness", "Crest Factor", "Estimated Speed"]（若有提及）,
  "tool": "tsa"、"fft"、"stft"（若提及則列出）
}}

### 範例1：
輸入：「請幫我對 Z 軸做 1500~4000Hz 濾波，並計算 crest factor」
→ 輸出：
{{
  "axis": "Z",
  "low_freq": 1500,
  "high_freq": 4000,
  "features": ["Crest Factor"]
}}

### 範例2：
輸入：「使用 TSA 對 X 軸進行分析，算 RMS 與轉速」
→ 輸出：
{{
  "axis": "X",
  "tool": "tsa",
  "features": ["RMS", "Estimated Speed"]
}}

### 使用者輸入：
{user_input}
"""
        messages = [{"role": "user", "content": prompt}]
        try:
            result = self.engine.generate(messages)
            return json.loads(result.strip().split("```")[0]) if "```" in result else json.loads(result)
        except:
            return {}
