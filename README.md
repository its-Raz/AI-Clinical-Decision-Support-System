# 🏥 Autonomous Clinical Decision Support System

An end-to-end, multi-agent clinical AI platform for safe medical triage, laboratory interpretation, dermatology screening, and evidence-based Q&A.

Built with **LangGraph**, **FastAPI**, **Streamlit**, **OpenAI**, **YOLOv8**, **Supabase**, and **Pinecone**, this system orchestrates specialized agents under a centralized clinical manager to deliver structured, explainable, and patient-friendly outputs.

---

# 📖 Overview

The **Autonomous Clinical Decision Support System** is a modular multi-agent architecture designed to:

- Triage incoming medical requests
- Interpret laboratory results
- Classify dermatological lesions from images
- Answer clinical questions using Retrieval-Augmented Generation
- Provide structured execution traces for transparency

The system emphasizes:

- Deterministic routing
- Tool-enforced reasoning
- Strict clinical guardrails
- Explainable execution steps
- Patient-friendly communication

---

# 🏗 System Architecture

## High-Level Flow

![Architecture](arch.png)

### Pipeline Stages

1. **User Input**
   - Text question and optionally image upload
   - Submitted via Streamlit frontend

2. **Semantic Router**
   - Uses OpenAI embeddings and cosine similarity
   - Instantly predicts category:
     - `blood_test_analysis`
     - `image_lesion_analysis`
     - `evidence_analyst`
     - `unsupported`
   - Returns score, confidence band, and spam rejection flag

3. **Manager Agent (Judge Node)**
   - LLM-based orchestrator
   - Reviews router proposal
   - Must call `judge_decision` tool
   - Either ACCEPTS or OVERRIDES routing

4. **Specialist Agent**
   - Blood Test Analyst
   - Skin Care Analyst
   - Evidence Analyst

5. **Deliver Node**
   - Converts clinical output into structured, empathetic patient message
   - Enforces formatting rules

6. **Response + Execution Trace**
   - Left panel: patient-friendly output
   - Right panel: step-by-step execution trace

---

# 🧠 Multi-Agent Design

## 1️⃣ Manager Agent

**Role:** Orchestrator and safety controller  

Prompt defined in:

`backend/prompts.py`

### Responsibilities

- Validate semantic router decisions  
- Call `judge_decision` tool to finalize classification  
- Route execution to the correct specialist agent  
- Format final structured patient message  
- Enforce strict workflow sequencing  

The Manager Agent does **not** perform medical reasoning.  
It supervises execution and guarantees safety boundaries.

---

## 2️⃣ Blood Test Analyst

**Architecture:** ReAct-style tool-constrained agent  

Prompts defined in:

`backend/agents/blood_test_analyst/prompts.py`

### Mandatory Tool Order

The agent enforces deterministic reasoning using tools:

1. `get_patient_history`  
2. `check_reference_range` (single call with all metrics)  
3. `search_medical_knowledge` (multiple short queries allowed)  
4. Generate structured clinical summary  

### Capabilities

- Retrieves patient demographics and lab history from Supabase  
- Validates all metrics against sex and age specific reference ranges  
- Detects abnormal trends  
- Correlates multi-metric abnormalities  
- Searches embedded medical knowledge for:
  - Causes
  - Diagnostic criteria
  - Workup recommendations
  - Treatment guidelines  

### Guardrails

- Always assume demo patient `P001` if none specified  
- Never call reference range tool multiple times  
- Keep RAG queries short, 2 to 5 words  
- No summary generation until all evidence gathered  

---

## 3️⃣ Skin Care Analyst

Prompts defined in:

`backend/agents/skin_care_analyst/prompts.py`

### Pipeline

1. YOLOv8 object detection  
2. Bounding box extraction  
3. Confidence scoring  
4. Urgency classification  
5. Structured dermatology summary  
6. Patient-friendly delivery message  

### Urgency Levels

- **High Urgency**
- **Low Urgency**

### Safety Rules

- Always state this is an AI screening  
- Never claim diagnosis  
- High urgency requires immediate dermatologist visit  
- Low urgency still recommends professional evaluation  

---

## 4️⃣ Evidence Analyst

Prompts defined in:

`backend/agents/evidence_analyst/prompts.py`

### Architecture

- Retrieval-Augmented Generation  
- Pinecone vector database  
- MedlinePlus embedded corpus  
- Maximum 3 retrieval queries per question  

### Strict Behavior

- Must base answer strictly on retrieved content  
- No hallucinations  
- If insufficient evidence found, disclose limitation honestly  

---

# ⚙ Backend Architecture

Core orchestration entry point:

`backend/main.py`

HTTP API layer:

`backend/api.py`

## Key Responsibilities

- Initialize semantic router once at startup  
- Initialize LangGraph system once  
- Build AgentState objects  
- Execute full multi-agent graph  
- Extract ordered execution steps  

---

# 🌐 API Endpoints

### GET `/api/team_info`

Returns metadata about team members.

---

### GET `/api/agent_info`

Returns:

- System description  
- Agent purposes  
- Prompt templates  
- Real example outputs  

---

### GET `/api/model_architecture`

Returns architecture diagram PNG.

---

### POST `/api/execute`

#### Request

```json
{
  "prompt": "Can you explain my blood test results?"
}

```json
{
  "final_response": "...",
  "steps": [
    { "module": "Semantic Router", ... },
    { "module": "Manager Agent", ... },
    { "module": "Blood Test Analyst", ... },
    { "module": "Deliver Node", ... }
  ]
}
