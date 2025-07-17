import re
import os
import base64
from pathlib import Path
from typing import List, Dict
from fireworks.client import Fireworks
from dotenv import load_dotenv

load_dotenv()

fireworks_api_key = os.getenv("FIREWORKS_API_KEY") 
if not fireworks_api_key:
    raise ValueError("FIREWORKS_API_KEY not found in environment variables")
else:
    llm = Fireworks(api_key=fireworks_api_key)

MODEL_NAME = "accounts/fireworks/models/qwen2p5-vl-32b-instruct"

def strip_model_thoughts(text: str) -> str:
    text = re.sub(r'<thought>(.*?)</thought>', '', text, flags=re.DOTALL)
    text = re.sub(r'<thinking>(.*?)</thinking>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>(.*?)</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<call:tool_code>.*?</call:tool_code>', '', text, flags=re.DOTALL)
    text = re.sub(r'<tool_code>.*?</tool_code>', '', text, flags=re.DOTALL)
    
    text = text.replace("<thought>", "").replace("</thought>", "")
    text = text.replace("<thinking>", "").replace("</thinking>", "")
    text = text.replace("<think>", "").replace("</think>", "")
    text = text.replace("<call:tool_code>", "").replace("</call:tool_code>", "")
    text = text.replace("<tool_code>", "").replace("</tool_code>", "")

    text = re.sub(r'^(Okay, |Alright, |Sure, |Here is the answer: |The answer is: |Based on the data, |I can help with that\. )', '', text, flags=re.IGNORECASE).strip()
    
    text = text.strip()
    text = re.sub(r'\n\s*\n', '\n', text)
    
    return text

def generate_answer(question: str, contexts: List[Dict]) -> str:
    prompt_text = (
        "You are a helpful RAG assistant specialized in answering questions about scientific PDFs. You will be given chunks of text and potentially images from a PDF and tables and diagrams. "
        "Use ONLY the provided context to answer the question accurately. Cite the source PDF and page where relevant.\n\n"
        "--- CONTEXT ---\n"
    )
    
    text_contexts_str = "\n\n".join([
        f"Source: {ctx['source_pdf']} page {ctx['page']}\n{ctx['text']}" for ctx in contexts
    ])
    
    prompt_text += text_contexts_str
    prompt_text += f"\n\n--- QUESTION ---\n{question}\n\n--- ANSWER ---\n"
    
    message_content = [{"type": "text", "text": prompt_text}]

    for ctx in contexts:
        if ctx.get("image_path"):
            img_path = Path(ctx["image_path"])
            if img_path.exists() and img_path.is_file():
                try:
                    with open(img_path, "rb") as f:
                        base64_image = base64.b64encode(f.read()).decode("utf-8")
                    
                    message_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}" 
                        },
                    })
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}") 

    try:
        response = llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[{
                "role": "user",
                "content": message_content,
            }],
            max_tokens=2048,
            temperature=0.1,
        )
        return strip_model_thoughts(response.choices[0].message.content)
    except Exception as e:
        return "Sorry, I was unable to generate an answer due to an API error."
