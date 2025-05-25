from fastapi import FastAPI, UploadFile, File, Form
from vibration_mcp_llm.vibration_analyze_tool import Tools
from vibration_mcp_llm.vibration_llm_agent import AIAgent
from vibration_mcp_llm.semantic_agent import SemanticAgent
import shutil, os, tempfile

app = FastAPI()
agent = AIAgent()
semantic_agent = SemanticAgent()

@app.post("/analyze_vibration/")
async def analyze(file: UploadFile = File(...), instruction_text: str = Form("")):
    # 儲存上傳檔案到暫存
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_path = tmp.name

    try:
        # 由語意代理人抽取指令（支援單一或多組）
        semantics = semantic_agent.extract_semantics(instruction_text)
        print("🧠 LLM 指令解析：", semantics)

        # 確保 semantics 一定是 list 格式
        if isinstance(semantics, dict):
            semantics_list = [semantics]
        elif isinstance(semantics, list):
            semantics_list = semantics
        else:
            semantics_list = []

        fs = 10240  # 預設取樣率

        # 一次批次執行所有指令組
        all_results = Tools.analyze_with_instructions(temp_path, fs, semantics_list)

        summary = agent.generate_summary(all_results)

    finally:
        os.remove(temp_path)

    return {
        "instructions": semantics_list,
        "results": all_results,
        "llm_summary": summary
    }
