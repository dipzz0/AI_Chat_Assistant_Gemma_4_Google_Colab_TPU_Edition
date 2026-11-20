"""
Google Colab X Gemma 4  AI Chat Assistant ( Google Colab TPU Edition)
High-performance conversational AI running natively on Google Cloud TPU.
"""

import os
import sys
import json
import time
from threading import Thread
import torch
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, AutoModelForImageTextToText, TextIteratorStreamer

# ==========================================================
# 1. Environment & Storage Setup
# ==========================================================
try:
    from google.colab import drive
    if not os.path.exists("/content/drive/MyDrive"):
        drive.mount('/content/drive')
    BASE_DIR = "/content/drive/MyDrive/Gemma_AI_Assistant"
    print("📁 Google Drive connected successfully!")
except Exception:
    BASE_DIR = os.path.join(os.getcwd(), "Gemma_AI_Assistant")
    print(f"⚠️ Using local workspace: {BASE_DIR}")

MODEL_DIR = os.path.join(BASE_DIR, "model_cache")
HISTORY_DIR = os.path.join(BASE_DIR, "chat_history")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

HISTORY_FILE = os.path.join(HISTORY_DIR, "active_chat.json")
MODEL_ID = "google/gemma-4-E4B-it"

# ==========================================================
# 2. Hardware TPU Verification
# ==========================================================
try:
    tpu_device = torch.device("tpu")
    _test = torch.ones(2, 2, device=tpu_device)
    print(f"✅ Google Cloud TPU Active! Verified Device: {_test.device}")
except Exception as e:
    print("❌ Google Cloud TPU not detected!")
    print("👉 Ensure you are running on a TPU-enabled environment.")
    sys.exit(1)

# ==========================================================
# 3. Model Download & Cache Verification
# ==========================================================
is_downloaded = os.path.exists(MODEL_DIR) and len(os.listdir(MODEL_DIR)) > 3

if is_downloaded:
    print(f"✅ Model '{MODEL_ID}' is already cached.")
else:
    print(f"📥 Downloading official {MODEL_ID} (Apache 2.0)...")
    print("Please wait, initial download takes approximately 2-4 minutes...")
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False
    )
    print("✅ Model downloaded and cached permanently.")

# ==========================================================
# 4. Stream Model Weights onto TPU
# ==========================================================
print("⏳ Loading Gemma 4 directly onto TPU (bfloat16)...")
processor = AutoProcessor.from_pretrained(MODEL_DIR)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_DIR,
    dtype=torch.bfloat16,
    device_map="tpu"
)
print("✅ Model successfully loaded on Google Cloud TPU!\n")

# ==========================================================
# 5. Persistent Memory Manager
# ==========================================================
class MemoryManager:
    def __init__(self, filepath=HISTORY_FILE):
        self.filepath = filepath
        self.history = self.load()

    def load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def save(self):
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

    def add_message(self, role, content):
        self.history.append({"role": role, "content": content})
        self.save()

    def clear(self):
        self.history = []
        self.save()
        return "🧹 Conversation memory cleared!"

    def get_messages(self):
        return self.history

memory = MemoryManager()

# ==========================================================
# 6. Real-Time Streaming Generation
# ==========================================================
def stream_chat(user_message):
    memory.add_message("user", user_message)

    formatted_prompt = processor.apply_chat_template(
        memory.get_messages(),
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(text=formatted_prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=250,
        do_sample=False,
        use_cache=True
    )

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    sys.stdout.write("🤖 Gemma 4: ")
    sys.stdout.flush()

    full_response = ""
    for chunk in streamer:
        sys.stdout.write(chunk)
        sys.stdout.flush()
        full_response += chunk

    thread.join()
    sys.stdout.write("\n\n" + "-" * 50 + "\n")
    sys.stdout.flush()

    memory.add_message("assistant", full_response.strip())
    return full_response.strip()

# ==========================================================
# 7. Interactive Terminal Loop
# ==========================================================
def main():
    print("=" * 60)
    print("🤖 Gemma 4 TPU Assistant CLI")
    print("💡 Commands: 'exit' to quit | 'clear' to reset conversation memory")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("👤 You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:
                print("\n👋 Session ended. Chat history is saved permanently.")
                break

            if user_input.lower() == "clear":
                memory.clear()
                print("🧹 Memory cleared! Starting a fresh conversation.\n")
                continue

            print()
            with torch.inference_mode():
                stream_chat(user_input)

        except KeyboardInterrupt:
            print("\n👋 Session stopped.")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")

if __name__ == "__main__":
    main()
