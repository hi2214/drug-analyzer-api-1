from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os

app = FastAPI()

HF_API_URL = "https://api-inference.huggingface.co/models/gpt2"  # 필요 시 다른 모델로 변경
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")  # Render에서 환경변수로 넣을 예정

headers = {"Authorization": f"Bearer {HF_API_TOKEN}"} if HF_API_TOKEN else {}

class MedInput(BaseModel):
    medications: list

@app.post("/analyze")
def analyze(data: MedInput):
    meds = ", ".join(data.medications)
    prompt = f"Check possible side effects and interactions between: {meds}"

    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 150}}
    resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
    if resp.status_code != 200:
        return {"error": f"Hugging Face API error {resp.status_code}: {resp.text}"}
    # HF Janssen / text-generation 모델들은 list 형태 반환
    j = resp.json()
    # 모델에 따라 반환 형식이 다르므로 가능한 안전하게 추출
    text = ""
    if isinstance(j, list) and "generated_text" in j[0]:
        text = j[0]["generated_text"]
    elif isinstance(j, dict) and "generated_text" in j:
        text = j["generated_text"]
    else:
        # 혹은 모델이 직접 텍스트를 줄 때
        if isinstance(j, list) and len(j)>0 and isinstance(j[0], str):
            text = j[0]
        else:
            text = str(j)

    return {"result": text}
