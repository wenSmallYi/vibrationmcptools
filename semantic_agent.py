import json
import re
from vibration_mcp_llm.llm_engine_endpoint_v2 import OpenAIEngine

def force_json_clean(raw: str) -> str:
    import re
    # 刪除 markdown ``` 標記
    raw = re.sub(r"```(?:json)?", "", raw)
    # 移除特殊標點
    raw = raw.replace('‘', '"').replace('’', '"').replace('“', '"').replace('”', '"')
    raw = raw.replace('「', '"').replace('」', '"')
    raw = raw.replace('，', ',').replace('、', ',')
    raw = raw.replace("'", '"')
    # 強制移除其他非可見ASCII
    raw = ''.join(c for c in raw if ord(c) < 128 or c in ['[', ']', '{', '}', ':', '"', ',', '.', '-', '_'])

    # **只保留第一個最長的 [...] 陣列**
    arr_match = re.search(r"\[.*\]", raw, flags=re.DOTALL)
    if arr_match:
        return arr_match.group(0).strip()
    # 如果不是陣列，也抓物件
    obj_match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if obj_match:
        return obj_match.group(0).strip()
    return raw.strip()

class SemanticAgent:
    def __init__(self):
        self.engine = OpenAIEngine()

    def extract_semantics(self, user_input: str) -> list:
        prompt = f"""
    請**只回傳乾淨JSON陣列**，不要任何說明與註解。格式例如：
    [
      {{
        "axis": ["X", "Y"],
        "processing": {{"method": "bandpass_filter", "params": {{"low": 500, "high": 4500}}}},
        "features": ["RMS", "Skewness", "Kurtosis", "CrestFactor"]
      }}
    ]
    ### 使用者輸入：
    {user_input}
    """
        messages = [{"role": "user", "content": prompt}]
        try:
            result = self.engine.generate(messages)
            json_input = force_json_clean(result)
            print("[DEBUG] force_json_clean結果:\n", json_input)
            parsed = json.loads(json_input)
            print("[DEBUG LLM 回傳內容]:\n", json.dumps(parsed, indent=2, ensure_ascii=False))
            return parsed
        except Exception as e:
            print("[SemanticAgent] JSON parse error:", e)
            print("=== force_json_clean 結果 ===\n", json_input)
            print("=== LLM 原始回傳 ===\n", result)
            return []