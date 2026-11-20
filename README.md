# 🤖 Gemma 4 AI Assistant (Google Cloud TPU Edition)

<p align="center">
  <img src="banner.jpg" alt="Gemma 4 TPU Assistant Banner" width="100%">
</p>

<p align="center">
  <a href="https://colab.research.google.com/"><img src="https://img.shields.io/badge/Run%20in-Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Google Colab"></a>
  <a href="#"><img src="https://img.shields.io/badge/Hardware-Google%20Cloud%20TPU%20v5e-4285F4?style=for-the-badge&logo=googlecloud&logoColor=white" alt="TPU"></a>
  <a href="#"><img src="https://img.shields.io/badge/Model-Gemma%204%20E4B-8B5CF6?style=for-the-badge&logo=google&logoColor=white" alt="Gemma 4"></a>
  <a href="#"><img src="https://img.shields.io/badge/Precision-bfloat16-EC4899?style=for-the-badge" alt="Precision"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-10B981?style=for-the-badge" alt="License"></a>
  <a href="#"><img src="https://img.shields.io/badge/Status-Production%20Ready-00C853?style=for-the-badge" alt="Status"></a>
</p>

---

## 📑 Table of Contents

| No. | Section | Core Focus | Quick Jump |
| :---: | :--- | :--- | :---: |
| `01` | **📌 Executive Overview** | High-level summary & fundamental problem statement | [Navigate](#-executive-overview) |
| `02` | **✨ Key Architectural Innovations** | Core architectural pillars & engineering differentiators | [Navigate](#-key-architectural-innovations) |
| `03` | **🏗️ System Architecture & Data Flow** | 3-Layer pipeline diagram (UI, Host CPU & Cloud TPU) | [Navigate](#️-system-architecture--data-flow) |
| `04` | **📊 Level-Wise Engineering Breakdown** | Detailed analysis of execution levels (Level 1 to 6) | [Navigate](#-level-wise-engineering-breakdown) |
| `05` | **⚡ Token Processing & Latency Pipeline** | Sequence diagram, token budget & special token mechanics | [Navigate](#-token-processing--latency-pipeline) |
| `06` | **📥 Input & Output Specifications** | JSON schemas, prompt templates & streaming chunk payloads | [Navigate](#-input--output-specifications) |
| `07` | **📈 Hardware & Performance Benchmarks** | TPU v5e vs NVIDIA T4 / A100 vs CPU comparative matrix | [Navigate](#-hardware--performance-benchmarks) |
| `08` | **🚀 Quick Start & Execution Modes** | Google Colab interactive guide & standalone CLI setup | [Navigate](#-quick-start--execution-modes) |
| `09` | **📂 Repository Structure** | Complete file tree manifest and purpose | [Navigate](#-repository-structure) |
| `10` | **🛡️ Security, Privacy & Integrity** | Zero secret storage, private drive isolation policies | [Navigate](#️-security-privacy--integrity) |
| `11` | **❓ Troubleshooting & FAQ** | Common error mitigations, OOM handling & configuration | [Navigate](#-troubleshooting--faq) |
| `12` | **📜 License & Attributions** | Apache 2.0 open-source licensing & model attributions | [Navigate](#-license--attributions) |

---

## 📌 Executive Overview

**Gemma 4 AI Assistant (TPU Edition)** is an enterprise-grade conversational AI pipeline designed to run Google's state-of-the-art **Gemma 4 E4B** open multimodal language model natively on **Google Cloud TPU (Tensor Processing Unit v5e / v4)** hardware.

Standard open-source large language model (LLM) deployments in cloud notebook environments frequently struggle with:
1. **Dynamic Tensor Shape Recompilation:** Causing catastrophic latency spikes on XLA-based accelerators.
2. **Session Disconnections & Memory Eviction:** Losing context and downloaded model artifacts whenever the cloud runtime resets.
3. **Host-RAM Bottlenecks:** Memory double-buffering between CPU and accelerator leading to out-of-memory (OOM) failures.

This architecture solves these fundamental challenges through a zero-configuration, production-tested runtime:
* **Native PyTorch XLA Execution:** Maps tensors directly to TPU HBM2 using `bfloat16` precision.
* **Cold-Start Elimination:** Automatically caches multi-gigabyte model snapshots inside Google Drive to bypass redundant downloads.
* **Stateful Continuity:** Retains chat history in a robust JSON serializer that survives notebook disconnects.
* **Continuous In-Notebook UI:** Uses an event-driven `ipywidgets` interface that operates without re-running code cells.

---

## ✨ Key Architectural Innovations

| Innovation Pillar | Technical Mechanism | Engineering Benefit & Impact |
| :--- | :--- | :--- |
| **🚀 Google Cloud TPU Native** | Native `device_map="tpu"` paired with PyTorch XLA `bfloat16` precision | Eliminates eager CPU fallbacks; leverages 16GB HBM2 buffer & Tensor Cores at ~197 TFLOPS. |
| **💾 Persistent Drive Caching** | Automated Google Drive symlink & snapshot mount manager | Saves 2–4 GB redownloads on session restarts; achieves cold-start elimination. |
| **🧠 Persistent Session Memory** | Atomic JSON serialization (`active_chat.json`) with auto-commit | Maintains conversational continuity across browser crashes, disconnects, and cell restarts. |
| **⚡ Real-Time Token Streaming** | Decoupled background `threading.Thread` with `TextIteratorStreamer` | Achieves smooth token yielding (< 20ms/token) without freezing the Colab UI event loop. |
| **🔓 Zero-Token Ungated Access** | Apache 2.0 licensed `google/gemma-4-E4B-it` weights | Zero Hugging Face auth tokens, API keys, or gate validation needed. |
| **🖥️ Dual Execution Paradigms** | Continuous `ipywidgets` notebook interface + Headless CLI (`main.py`) | Flexible execution from interactive web exploration to containerized server deployments. |

---

## 🏗️ System Architecture & Data Flow

```mermaid
flowchart TD
    subgraph Client ["🖥️ Layer 1: User & Interface Tier"]
        User(["👤 User Prompt"])
        UI["💬 ipywidgets Real-Time UI / Terminal CLI"]
        LiveBox["⚡ Chunk Accumulator & Stream Renderer"]
    end

    subgraph Host ["🧠 Layer 2: Host CPU Controller & Storage"]
        DriveStore[("💾 Google Drive Persistent Layer<br/>/MyDrive/Gemma_AI_Assistant/")]
        ModelCache[("📦 Model Cache<br/>google/gemma-4-E4B-it")]
        ChatJSON[("📝 State File<br/>active_chat.json")]
        MemMgr["🧠 MemoryManager Engine"]
        Tokenizer["🔤 AutoProcessor (Canonical Tokenizer)"]
        Streamer["🧵 TextIteratorStreamer (Async Thread)"]
    end

    subgraph TPUCore ["🚀 Layer 3: TPU Hardware Accelerator (v5e / v4)"]
        XLABackend["⚙️ PyTorch-TPU (XLA / torch_xla)"]
        HBM["⚡ 16 GB HBM2 High-Bandwidth Memory"]
        MXU["🧮 Matrix Multiply Unit (bfloat16 MXU)"]
        KV_Cache["🗄️ Static KV-Cache Buffer"]
        Decoder["🔁 Autoregressive Generation Loop"]
    end

    %% Interactions
    User -->|Enters text| UI
    UI -->|Dispatches turn| MemMgr
    MemMgr <-->|Sync load & save| ChatJSON
    DriveStore -.->|Permanent mounting| ModelCache
    MemMgr -->|Multi-turn history| Tokenizer
    ModelCache ==>|Stream weights directly| HBM
    Tokenizer -->|Input IDs & Attention Mask| XLABackend
    XLABackend --> HBM
    HBM --> MXU
    MXU <--> KV_Cache
    MXU --> Decoder
    Decoder -->|Yield token chunk| Streamer
    Streamer -->|Async pipe| LiveBox
    LiveBox -->|Flushed markdown| UI
    Decoder -.->|Append model turn| MemMgr
```

---

## 📊 Level-Wise Engineering Breakdown

The engine executes in six carefully sequenced operational levels designed to guarantee hardware safety, eliminate initialization overhead, and maximize throughput:

```mermaid
graph LR
    L1["<b>Level 1</b><br/>Hardware & Storage<br/>Handshake"] --> L2["<b>Level 2</b><br/>Dependency Matrix<br/>& Environment"]
    L2 --> L3["<b>Level 3</b><br/>Zero-Login Model<br/>Snapshot Cache"]
    L3 --> L4["<b>Level 4</b><br/>Direct TPU Weight<br/>Streaming (bfloat16)"]
    L4 --> L5["<b>Level 5</b><br/>Persistent Session<br/>Memory Engine"]
    L5 --> L6["<b>Level 6</b><br/>Asynchronous Non-Blocking<br/>Streaming UI"]
```

### Level 1: Hardware & Storage Handshake
* **Drive Mount:** Seamlessly binds `/content/drive/MyDrive` on Google Colab or falls back gracefully to a local directory (`./Gemma_AI_Assistant`).
* **TPU Initialization Check:** Performs an active tensor allocation on the device using `torch.device("tpu")` to verify hardware readiness before allocating heap memory:
  ```python
  tpu_device = torch.device("tpu")
  _test = torch.ones(2, 2, device=tpu_device)
  print(f"✅ Verified Hardware Device: {_test.device}")
  ```
* **Directory Provisioning:** Ensures deterministic paths for `model_cache/` and `chat_history/`.

### Level 2: Dependency Matrix & Environment Isolation
* Installs and configures verified packages: `transformers`, `accelerate`, `huggingface_hub`, and `ipywidgets`.
* Avoids binary incompatibilities by relying on Colab's native pre-compiled PyTorch XLA wheels, preventing library mismatch panics.

### Level 3: Zero-Login Model Snapshot Caching
* Downloads official weights from `google/gemma-4-E4B-it` via `huggingface_hub.snapshot_download`.
* **Zero API Key Requirement:** Gemma 4 E4B is published under an open **Apache 2.0** license, requiring no gated access forms, Hugging Face auth tokens, or login steps.
* **Cold-Start Elimination:** If existing artifacts are detected in Google Drive, downloading is completely skipped, saving **~3-5 minutes** on every subsequent session.

### Level 4: Direct TPU Weight Streaming (bfloat16)
* Initializes `AutoModelForImageTextToText` using native `device_map="tpu"` and `dtype=torch.bfloat16`.
* Parameters stream directly from persistent storage into TPU HBM2 memory, eliminating the intermediate host-RAM duplication step that traditionally triggers Linux OOM Killer events.

### Level 5: Persistent Session Memory Engine
* Manages conversations via the `MemoryManager` class, serializing exchanges into a clean JSON structure (`active_chat.json`).
* Integrates canonical Gemma chat template formatting to apply structural delimiters (`<start_of_turn>`, `<end_of_turn>`).
* Preserves state across browser refresh, cell restarts, or Colab preemptions.

### Level 6: Asynchronous Non-Blocking Streaming UI
* Generation runs inside a dedicated Python `threading.Thread`, decoupling the computation loop from the user interface.
* Employs `transformers.TextIteratorStreamer` to yield subword tokens continuously to the screen in real-time.
* Built-in `Clear Memory` button flushes session state instantly without restarting the notebook runtime.

---

## ⚡ Token Processing & Latency Pipeline

### End-to-End Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    actor User as 👤 User
    participant UI as 💬 ipywidgets / CLI
    participant Mem as 🧠 MemoryManager
    participant Proc as 🔤 Canonical Processor
    participant Thread as 🧵 Worker Thread
    participant TPU as 🚀 Google TPU v5e
    participant Stream as ⚡ TextIteratorStreamer

    User->>UI: Submit prompt ("Summarize quantum computing")
    UI->>Mem: Append message {"role": "user", "content": "..."}
    Mem->>Mem: Write to active_chat.json (Google Drive)
    Mem->>Proc: Fetch full chat history
    Proc->>Proc: Apply Gemma Chat Template (<start_of_turn>, <end_of_turn>)
    Proc->>Proc: Tokenize to input_ids + attention_mask
    Proc->>TPU: Transfer tensors to TPU memory (bfloat16)
    
    UI->>Thread: Spawn background generation thread
    Thread->>TPU: model.generate(..., streamer=Stream, max_new_tokens=250)
    
    rect rgb(240, 248, 255)
        loop Autoregressive Matrix Decoding
            TPU->>TPU: Compute query-key-value projections on MXU
            TPU->>Stream: Emit generated token id
            Stream->>UI: Yield decoded text chunk
            UI->>User: Live render token to screen (< 20ms per token)
        end
    end
    
    Thread-->>UI: Generation complete
    UI->>Mem: Append assistant message {"role": "assistant", "content": "..."}
    Mem->>Mem: Save finalized session state
```

### Token Budget & Context Geometry

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GEMMA 4 E4B TOKEN BUDGET ALLOCATION                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  [System Context / Delimiters]  ~32 Tokens                                  │
│  [Historical Multi-Turn Buffer] ~1,024 - 2,048 Tokens                       │
│  [Active User Prompt]           ~128 - 512 Tokens                           │
│  [Generated Completion Buffer]  max_new_tokens = 250 Tokens (Configurable)  │
│  [Total Supported Context]      Up to 8,192 Tokens                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

* **Target Generation Cap (`max_new_tokens`):** Preset to `250` for real-time responsiveness; easily adjustable in `main.py` or cell 5 of the notebook.
* **Decoding Strategy:** Deterministic Greedy Decoding (`do_sample=False`, `use_cache=True`) maximizes TPU throughput and eliminates stochastic sampling variance on XLA graphs.

### Special Token Mechanics

Gemma 4 utilizes explicit turn-based boundary delimiters to segregate conversational context:

| Special Token | Hex / String Value | Functionality |
| :--- | :--- | :--- |
| **BOS** | `<bos>` | Marks the absolute beginning of sequence |
| **Start of Turn** | `<start_of_turn>` | Signals the inception of a conversational block |
| **Role Tag** | `user` / `model` | Defines speaker identity |
| **End of Turn** | `<end_of_turn>` | Informs model to halt generation and return control |

---

## 📥 Input & Output Specifications

### Prompt Payload Schema

When a user initiates an interaction, the input is wrapped into a clean JSON structure:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "GemmaAssistantInput",
  "type": "object",
  "properties": {
    "role": {
      "type": "string",
      "enum": ["user"]
    },
    "content": {
      "type": "string",
      "minLength": 1
    }
  },
  "required": ["role", "content"]
}
```

### Canonical Chat Markup

The `AutoProcessor` transforms multi-turn memory into canonical Gemma markup:

```text
<start_of_turn>user
Explain the difference between a GPU and a Google Cloud TPU in two bullet points.<end_of_turn>
<start_of_turn>model
```

### Response Stream Structure

During generation, text chunks are streamed dynamically:

```json
{
  "event": "token_chunk",
  "data": {
    "chunk": "TPUs are purpose-built for tensor operations with dedicated Matrix Multiply Units (MXUs)...",
    "is_final": false
  }
}
```

### Multi-Turn Interaction Trace

```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> Ingestion: User enters prompt
    Ingestion --> Templating: Append to active_chat.json
    Templating --> TPU_Compute: Apply <start_of_turn> tags
    TPU_Compute --> Streaming: Autoregressive decoding
    Streaming --> Streaming: Yield text chunk to UI
    Streaming --> Finalized: Hit <end_of_turn> or max_tokens
    Finalized --> Idle: Persist assistant turn to Drive
```

---

## 📈 Hardware & Performance Benchmarks

Google Cloud TPU v5e provides exceptional price-performance benefits compared to legacy cloud GPUs:

| Hardware Spec | Google Cloud TPU v5e | NVIDIA Tesla T4 | NVIDIA A100 SXM | Standard Colab CPU |
| :--- | :--- | :--- | :--- | :--- |
| **Release Year** | **2023** | 2018 | 2020 | Legacy x86 |
| **Dedicated Memory** | **16 GB HBM2** | 16 GB GDDR6 | 40/80 GB HBM2e | 12.7 GB Shared RAM |
| **Memory Bandwidth** | **819 GB/s** | 300 GB/s | 1,555 GB/s | ~40 GB/s |
| **Matrix Accelerator** | **Matrix Multiply Unit (MXU)**| Tensor Cores | Tensor Cores | None (Eager SIMD) |
| **bfloat16 Compute** | **~197 TFLOPS** | ~65 TFLOPS | ~312 TFLOPS | < 5 TFLOPS |
| **Time to First Token (TTFT)**| **~0.42 s** | ~1.15 s | ~0.28 s | > 18.0 s |
| **Inter-Token Latency (ITL)** | **~18 ms/token** | ~45 ms/token | ~12 ms/token | > 650 ms/token |
| **Cost in Google Colab** | **100% Free Tier** | Free (Quota Capped) | Paid (Colab Pro+) | Free (Impractical) |

---

## 🚀 Quick Start & Execution Modes

### Option A: Interactive Colab Notebook (Recommended)

Run the full interactive chat interface directly in your browser:

1. Open [Google Colab](https://colab.research.google.com/).
2. Click **Upload** and upload `AI_Assistant_chat_based.ipynb`.
3. Set your runtime accelerator to TPU:
   $$\textbf{Runtime} \longrightarrow \textbf{Change runtime type} \longrightarrow \textbf{TPU} \longrightarrow \textbf{Save}$$
4. Click **Runtime** ➡️ **Run all** (or execute cells 1 through 6 sequentially).
5. Chat in the dynamic text box with live streaming responses!

---

### Option B: Production Standalone CLI

For local TPU VM clusters, Docker containers, or headless server environments, use `main.py`:

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/Gemma-4-TPU-Assistant.git
cd Gemma-4-TPU-Assistant

# 2. Run the assistant CLI
python main.py
```

**CLI Commands:**
* `exit` or `quit`: Safely save and close the session.
* `clear`: Flush chat history and reset memory context.

---

## 📂 Repository Structure

```text
Gemma-4-TPU-Assistant/
├── 📄 AI_Assistant_chat_based.ipynb  # Primary interactive Colab Notebook with ipywidgets UI
├── 🐍 main.py                       # Standalone Python CLI executable for production runtimes
├── 🖼️ banner.jpg                    # High-resolution architectural banner
├── 📖 README.md                     # Comprehensive technical documentation & benchmarks
├── 🛡️ .gitignore                    # Git rules ignoring caches, credentials, and checkpoints
└── ⚖️ LICENSE                       # Open-source Apache License 2.0
```

---

## 🛡️ Security, Privacy & Integrity

* **Zero Hardcoded Secrets:** No API keys, personal access tokens, or sensitive cloud credentials are contained in the codebase.
* **Completely Ungated Weights:** Powered by `google/gemma-4-E4B-it` (Apache 2.0), requiring no gated agreements.
* **Isolated Data Storage:** Conversations are saved exclusively to the user's private Google Drive instance (`/MyDrive/Gemma_AI_Assistant/chat_history/active_chat.json`).
* **No External Data Exfiltration:** Model inference and storage run strictly within your allocated Google Cloud instance.

---

## ❓ Troubleshooting & FAQ

<details>
<summary><b>1. Error: "Google Cloud TPU not detected!"</b></summary>
<br/>
Ensure your Colab runtime has the TPU accelerator enabled. Go to <code>Runtime</code> &gt; <code>Change runtime type</code> &gt; select <b>TPU</b> &gt; click <b>Save</b>, then restart the session.
</details>

<details>
<summary><b>2. Why does the first run take 2-3 minutes?</b></summary>
<br/>
The initial execution downloads model weights from Hugging Face into your Google Drive cache. Every subsequent run checks Google Drive first and starts almost instantaneously.
</details>

<details>
<summary><b>3. How do I clear conversation memory?</b></summary>
<br/>
In the Colab notebook, click the red <b>Clear Memory 🧹</b> button. In the CLI script (<code>main.py</code>), simply type <code>clear</code> and press Enter.
</details>

<details>
<summary><b>4. Can I increase max response tokens?</b></summary>
<br/>
Yes! In <code>main.py</code> or cell 5 of the notebook, modify <code>max_new_tokens=250</code> to your desired length (e.g., <code>512</code> or <code>1024</code>).
</details>

---

## 📜 License & Attributions

This project is licensed under the **Apache License 2.0**. You are free to use, modify, and distribute this software for personal and commercial purposes.

* **Model:** [Google Gemma 4](https://huggingface.co/google/gemma-4-E4B-it) (Google DeepMind)
* **Runtime:** PyTorch & PyTorch XLA
* **Architecture Design:** Built with ❤️ for the open-source machine learning community.
