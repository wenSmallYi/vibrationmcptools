from fastapi import FastAPI, UploadFile, File, Form
from vibration_analyze_tool import Tools
from semantic_agent import SemanticAgent
from instruction_builder import MCPInstructionBuilder
from vibration_llm_agent import AIAgent
import shutil, os, tempfile

app = FastAPI()
semantic_agent = SemanticAgent()
agent = AIAgent()

@app.post("/analyze_vibration/")
async def analyze(file: UploadFile = File(...), instruction_text: str = Form("")):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_path = tmp.name

    try:
        semantics = semantic_agent.extract_semantics(instruction_text)
        final_instr = MCPInstructionBuilder.build(semantics)

        print("🧠 使用者輸入指令：", instruction_text)
        print("🔎 語意抽取結果：", semantics)
        print("⚙️ 最終執行指令：", final_instr)

        features = Tools.analyze_with_instruction(temp_path, fs=10240, instruction=final_instr)
        summary = agent.generate_summary(features)
    finally:
        os.remove(temp_path)

    return {
        "instruction": final_instr,
        "features": features,
        "llm_summary": summary
    }