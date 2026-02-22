"""
run_evaluation.py — Evaluation Pipeline Execution Script.

Orchestrates the full evaluation loop:
  1. Loads the golden dataset (eval_dataset.py)
  2. Constructs AgentState for each example
  3. Invokes the compiled LangGraph pipeline (system.run())
  4. Runs all 6 LLM-as-a-Judge evaluators per result
  5. Aggregates scores per category and overall
  6. Pushes all runs, inputs, outputs, and feedback to LangSmith

Evaluators applied per example:
  routing_accuracy    → ALL examples
  safety_robustness   → Category B + C where should_refuse=True
  clinical_tone       → Category A only
  tool_selection      → ALL examples
  tool_sequencing     → ALL examples
  task_completeness   → ALL examples

Usage:
  python run_evaluation.py

Required environment variables (from your .env):
  OPENAI_API_KEY     — used by the clinical system and the judge LLM
  LANGSMITH_API_KEY  — LangSmith API key
  LANGSMITH_ENDPOINT — LangSmith endpoint URL
  LANGSMITH_PROJECT  — LangSmith project name

Optional:
  EVAL_JUDGE_MODEL      — judge LLM model, defaults to "gpt-4o"
  SKIP_REAL_PIPELINE    — "true" to mock the pipeline (harness unit testing)
  EVAL_SLEEP_SECONDS    — sleep between runs, default 1.0
"""

import json
import logging
import os
import signal
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Map your .env LANGSMITH_* names to the LANGCHAIN_* names that the
# LangChain tracing layer reads internally. Must be set before any
# langchain / langsmith imports that trigger tracer initialisation.
# ─────────────────────────────────────────────────────────────────────────────

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"]    = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_ENDPOINT"]   = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGCHAIN_PROJECT"]    = os.getenv("LANGSMITH_PROJECT", "Clinical-System-Eval-v1")

from langsmith import Client as LangSmithClient   # noqa: E402
from langsmith.evaluation import evaluate          # noqa: E402

from eval_dataset import EVAL_DATASET              # noqa: E402

# ── Six evaluators, one file each ────────────────────────────────────────────
from eval_judge_routing           import RoutingAccuracyEvaluator    # noqa: E402
from eval_judge_safety            import SafetyRobustnessEvaluator   # noqa: E402
from eval_judge_tone              import ClinicalToneEvaluator        # noqa: E402
from eval_judge_tool_selection    import ToolSelectionEvaluator       # noqa: E402
from eval_judge_tool_sequencing   import ToolSequencingEvaluator      # noqa: E402
from eval_judge_task_completeness import TaskCompletenessEvaluator    # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("eval_runner")

SKIP_REAL_PIPELINE = os.getenv("SKIP_REAL_PIPELINE", "false").lower() == "true"
DATASET_NAME       = "clinical-system-golden-v1"
PROJECT_NAME       = os.getenv("LANGSMITH_PROJECT", "Clinical-System-Eval-v1")
SLEEP_BETWEEN_RUNS = float(os.getenv("EVAL_SLEEP_SECONDS", "1.0"))

ALL_EVAL_KEYS = [
    "routing_accuracy",
    "safety_robustness",
    "clinical_tone",
    "tool_selection",
    "tool_sequencing",
    "task_completeness",
]

# ─────────────────────────────────────────────────────────────────────────────
# Graceful stop flag
#
# WHY: langsmith.evaluate() spawns a ThreadPoolExecutor internally.
# Killing the terminal (Ctrl+C / SIGTERM) only signals the main thread;
# the worker threads and LangSmith's background trace uploader keep
# running — that's why experiments keep appearing in LangSmith after
# the terminal is closed.
#
# HOW: We register signal handlers that set _STOP_EVENT. The pipeline
# target checks this flag before each run and returns a sentinel result
# immediately when it's set. The thread pool drains in < 1 second instead
# of continuing for the full 40-example run.
# ─────────────────────────────────────────────────────────────────────────────

_STOP_EVENT = threading.Event()


def _handle_stop(signum, frame):
    if _STOP_EVENT.is_set():
        # Second Ctrl+C: user wants immediate kill
        log.warning("⛔  Second stop signal — force-killing immediately.")
        os._exit(1)

    _STOP_EVENT.set()
    log.warning(
        "\n⛔  Stop signal received. Flushing LangSmith traces (3s), then exiting.\n"
        "   Press Ctrl+C again to kill immediately."
    )

    # WHY os._exit via a timer thread:
    # langsmith.evaluate() pre-submits ALL 40 examples to its ThreadPoolExecutor
    # at startup. Setting the stop event makes each target() return instantly
    # as CANCELLED, but evaluate() still processes all 40 results and keeps
    # uploading to LangSmith. The only way to truly stop it is to kill the
    # process. We wait 3 seconds to let the current in-flight trace flush.
    def _force_exit():
        time.sleep(3)
        log.warning("⛔  Force-exiting (3s elapsed).")
        os._exit(0)

    threading.Thread(target=_force_exit, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# State builder
# ─────────────────────────────────────────────────────────────────────────────

def build_state_from_example(example: dict) -> dict:
    """
    Translate a golden dataset example into a fully-formed AgentState.

    Router confidence strategy:
      Category A → high confidence (0.91) — we want to test pipeline quality,
                   not whether the Judge can override a bad router signal.
      Category B/C → medium confidence (0.48) — deliberately weak signal to
                     stress-test the Judge's override capability.
    """
    cat = example.get("category", "A")

    if cat == "A":
        router_score      = 0.91
        router_confidence = "high"
        sub = example.get("subcategory", "")
        if "blood_test" in sub:
            proposed = "blood_test_analysis"
        elif "skin" in sub:
            proposed = "image_lesion_analysis"
        else:
            proposed = "evidence_analyst"
    else:
        router_score      = 0.48
        router_confidence = "medium"
        proposed          = "unsupported"

    return {
        "raw_user_input":            example["input_text"],
        "router_proposed_category":  proposed,
        "router_score":              router_score,
        "router_confidence":         router_confidence,
        "request_type":              "",
        "patient_id":                f"EVAL-{example['id']}",
        "lab_result":                example.get("lab_result"),
        "lab_insights":              None,
        "image_path":                example.get("image_path"),
        "vision_results":            None,
        "vision_insights":           None,
        "evidence_insights":         None,
        "messages":                  [],
        "next_step":                 "",
        "final_report":              None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Mock pipeline (SKIP_REAL_PIPELINE=true — zero API cost harness test)
# ─────────────────────────────────────────────────────────────────────────────

def _mock_output(state: dict, example: dict) -> dict:
    """Return a structurally valid AgentState without calling the real system."""
    expected_route = example.get("expected_route", "deliver")
    should_refuse  = example.get("should_refuse", False)
    pid            = state["patient_id"]

    def _judge_trace(accepted: str, override: bool = False):
        return {
            "role": "system",
            "content": (
                f"[Judge] Patient {pid} | "
                f"Router proposed: '{state['router_proposed_category']}' "
                f"({state['router_score']:.4f}, {state['router_confidence']}) | "
                f"Judge accepted: '{accepted}' | "
                f"Overridden: {override} | "
                f"Reason: Mock evaluation run."
            ),
        }

    def _deliver_trace():
        return {"role": "system",
                "content": f"[Manager → deliver_node] Final patient message generated for {pid}."}

    if should_refuse:
        return {**state,
            "request_type": "unsupported",
            "next_step":    "deliver",
            "final_report": (
                "I'm sorry, but I'm unable to assist with that request. "
                "As a clinical assistant, I can only help with medical questions, "
                "lab results, or skin lesion analysis. Please rephrase your query."
            ),
            "messages": [_judge_trace("unsupported"), _deliver_trace()],
        }

    if "blood_test" in expected_route:
        return {**state,
            "request_type": "blood_test_analysis",
            "next_step":    "blood_test_analyst",
            "lab_insights": (
                "Glucose: 112 mg/dL — elevated (ref 70-100). "
                "Hemoglobin: 11.8 g/dL — low (ref 12.0-15.5). "
                "Creatinine: 1.4 mg/dL — elevated (ref 0.6-1.2). "
                "Recommend GP follow-up within 2 weeks."
            ),
            "final_report": (
                "Hi! Your latest blood test results have arrived.\n\n"
                "### Glucose\nYour glucose is slightly above normal at 112 mg/dL "
                "(normal: 70-100). This may indicate pre-diabetic tendencies.\n\n"
                "### Hemoglobin\nYour hemoglobin is 11.8 g/dL — a bit low, which "
                "can cause fatigue. Your doctor may want to investigate further.\n\n"
                "### Creatinine\nYour creatinine is 1.4 mg/dL — mildly elevated. "
                "This is a kidney function marker worth discussing with your GP.\n\n"
                "### Next Steps\nPlease schedule a follow-up within 2 weeks."
            ),
            "messages": [
                _judge_trace("blood_test_analysis"),
                {"role": "system", "content": "[blood_test_analyst] Analysis complete."},
                {"role": "assistant", "content": "Blood test report generated."},
                _deliver_trace(),
            ],
        }

    if "skin" in expected_route:
        return {**state,
            "request_type":    "image_lesion_analysis",
            "next_step":       "skin_care_analyst",
            "vision_insights": (
                "YOLOv11 inference on ISIC_0024323.jpg: Label=MEL (melanoma-like), "
                "Confidence=0.87, BBox=[120,85,340,270]. High-urgency lesion detected."
            ),
            "final_report": (
                "Hi! We've completed the preliminary analysis of your skin lesion.\n\n"
                "## AI Screening Results\n"
                "- **What we detected**: An irregular pigmented skin change\n"
                "- **Urgency level**: High Urgency\n"
                "- **AI confidence**: 87%\n\n"
                "## What should you do next?\n"
                "Please see a board-certified dermatologist within 24-48 hours.\n\n"
                "## Important Reminder\n"
                "This AI screening is a preliminary assessment tool, not a medical diagnosis."
            ),
            "messages": [
                _judge_trace("image_lesion_analysis"),
                {"role": "system", "content": "[skin_care_analyst] Vision analysis complete."},
                {"role": "assistant", "content": "Skin lesion report generated."},
                _deliver_trace(),
            ],
        }

    # Default: evidence_analyst
    return {**state,
        "request_type":      "evidence_analyst",
        "next_step":         "evidence_analyst",
        "evidence_insights": (
            "RAG result from MedlinePlus/Pinecone: Metformin commonly causes GI "
            "side effects including nausea, diarrhoea, and abdominal discomfort. "
            "Rare but serious: lactic acidosis, especially in renal impairment. "
            "Source: MedlinePlus Drug Information."
        ),
        "final_report": (
            "Hi! I have some information regarding your medical question.\n\n"
            "Based on clinical evidence, Metformin commonly causes gastrointestinal "
            "side effects — nausea, diarrhoea, and stomach discomfort — especially "
            "when first starting. These usually improve over time.\n\n"
            "In rare cases, a serious condition called lactic acidosis can occur, "
            "particularly in people with kidney problems.\n\n"
            "Please consult your doctor or pharmacist for personalised guidance. "
            "This information is for educational purposes only."
        ),
        "messages": [
            _judge_trace("evidence_analyst"),
            {"role": "system", "content": "[evidence_analyst] RAG retrieval complete."},
            {"role": "assistant", "content": "Evidence report generated."},
            _deliver_trace(),
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# LangSmith dataset management
# ─────────────────────────────────────────────────────────────────────────────

def create_or_update_langsmith_dataset(client: LangSmithClient) -> str:
    """Upsert the golden dataset into LangSmith. Returns the dataset ID."""
    datasets = list(client.list_datasets(dataset_name=DATASET_NAME))
    if datasets:
        log.info("Reusing existing dataset: %s (id=%s)", DATASET_NAME, datasets[0].id)
        return str(datasets[0].id)

    log.info("Creating new LangSmith dataset: %s", DATASET_NAME)
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description=(
            "Clinical Multi-Agent System — Golden Evaluation Dataset v2. "
            "40 examples across 3 categories. "
            "Evaluators: routing_accuracy, safety_robustness, clinical_tone, "
            "tool_selection, tool_sequencing, task_completeness."
        ),
    )

    inputs_list, outputs_list = [], []

    for ex in EVAL_DATASET:
        state = build_state_from_example(ex)
        inputs_list.append({
            "raw_user_input":           state["raw_user_input"],
            "router_proposed_category": state["router_proposed_category"],
            "router_score":             state["router_score"],
            "router_confidence":        state["router_confidence"],
            "patient_id":               state["patient_id"],
            "lab_result":               state.get("lab_result"),
            "image_path":               state.get("image_path"),
        })
        outputs_list.append({
            "expected_route":      ex["expected_route"],
            "expected_behavior":   ex["expected_behavior"],
            "should_refuse":       ex["should_refuse"],
            "requires_specialist": ex["requires_specialist"],
            "category":            ex["category"],
            "subcategory":         ex["subcategory"],
            "example_id":          ex["id"],
            "expected_tools":      ex["expected_tools"],
            "forbidden_tools":     ex["forbidden_tools"],
            "completion_criteria": ex["completion_criteria"],
        })

    client.create_examples(
        dataset_id=str(dataset.id),
        inputs=inputs_list,
        outputs=outputs_list,
    )
    log.info("Uploaded %d examples.", len(EVAL_DATASET))
    return str(dataset.id)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline target factory
# ─────────────────────────────────────────────────────────────────────────────

def make_pipeline_target(system, example_lookup: dict):
    """
    Returns target(inputs: dict) -> dict for langsmith.evaluate().

    Checks _STOP_EVENT before every run so Ctrl+C drains the thread pool
    within one example's latency instead of running to completion.
    """
    def _target(inputs: dict) -> dict:
        # ── Early-exit when stop was requested ───────────────────────────────
        if _STOP_EVENT.is_set():
            patient_id = inputs.get("patient_id", "unknown")
            log.info("⛔  Skipping %s — stop event is set.", patient_id)
            return {
                "raw_user_input":           inputs.get("raw_user_input", ""),
                "router_proposed_category": inputs.get("router_proposed_category", ""),
                "router_score":             inputs.get("router_score", 0.0),
                "router_confidence":        inputs.get("router_confidence", ""),
                "request_type":             "cancelled",
                "patient_id":               patient_id,
                "lab_result":               None,
                "lab_insights":             None,
                "image_path":               None,
                "vision_results":           None,
                "vision_insights":          None,
                "evidence_insights":        None,
                "messages":                 [],
                "next_step":                "cancelled",
                "final_report":             "[EVAL CANCELLED BY USER]",
            }

        patient_id = inputs.get("patient_id", "")
        example    = example_lookup.get(patient_id, {})

        state = {
            "raw_user_input":            inputs.get("raw_user_input", ""),
            "router_proposed_category":  inputs.get("router_proposed_category", ""),
            "router_score":              inputs.get("router_score", 0.5),
            "router_confidence":         inputs.get("router_confidence", "medium"),
            "request_type":              "",
            "patient_id":                patient_id,
            "lab_result":                inputs.get("lab_result"),
            "lab_insights":              None,
            "image_path":                inputs.get("image_path"),
            "vision_results":            None,
            "vision_insights":           None,
            "evidence_insights":         None,
            "messages":                  [],
            "next_step":                 "",
            "final_report":              None,
        }

        if SKIP_REAL_PIPELINE:
            log.info("[MOCK] patient_id=%s", patient_id)
            return _mock_output(state, example)

        try:
            log.info("Running pipeline for %s", patient_id)
            result = system.run(state)
            time.sleep(SLEEP_BETWEEN_RUNS)
            return result
        except Exception as e:
            log.error("Pipeline error [%s]: %s", patient_id, e)
            return {**state, "final_report": f"[PIPELINE ERROR] {e}",
                    "request_type": "error", "next_step": "error"}

    return _target


# ─────────────────────────────────────────────────────────────────────────────
# Metrics aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_results(results_list: list) -> dict:
    """Compute per-category and overall means. score=None values are excluded."""
    cat_by_patient = {f"EVAL-{ex['id']}": ex["category"] for ex in EVAL_DATASET}

    per_cat = defaultdict(lambda: defaultdict(list))
    overall = defaultdict(list)

    for result in results_list:
        run_output = result.get("run", {}) or {}
        patient_id = (run_output.get("inputs") or {}).get("patient_id", "unknown")
        category   = cat_by_patient.get(patient_id, "unknown")

        for key, stats in (result.get("feedback_stats") or {}).items():
            score = stats.get("avg")
            if score is None:
                continue
            per_cat[category][key].append(score)
            overall[key].append(score)

    def _mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else None

    metrics = {
        "per_category": {cat: {k: _mean(v) for k, v in keys.items()}
                         for cat, keys in per_cat.items()},
        "overall":      {k: _mean(v) for k, v in overall.items()},
    }
    valid = [v for v in metrics["overall"].values() if v is not None]
    metrics["composite_score"] = round(sum(valid) / len(valid), 4) if valid else None
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Terminal report
# ─────────────────────────────────────────────────────────────────────────────

def print_report(metrics: dict) -> None:
    W = 72
    eval_labels = {
        "routing_accuracy":  "Routing Accuracy",
        "safety_robustness": "Safety Robustness",
        "clinical_tone":     "Clinical Tone & Synthesis",
        "tool_selection":    "Tool Selection",
        "tool_sequencing":   "Tool Sequencing",
        "task_completeness": "Task Completeness",
    }
    cat_labels = {
        "A": "Category A — Core Functional",
        "B": "Category B — Security & Adversarial",
        "C": "Category C — OOD & Noise",
    }

    print("\n" + "═" * W)
    print("  CLINICAL SYSTEM EVALUATION REPORT")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  {PROJECT_NAME}")
    print("═" * W)

    for cat in ["A", "B", "C"]:
        cat_metrics = metrics["per_category"].get(cat, {})
        print(f"\n  {cat_labels[cat]}")
        print("  " + "─" * (W - 2))
        for k in ALL_EVAL_KEYS:
            v = cat_metrics.get(k)
            label = eval_labels.get(k, k)
            if v is None:
                print(f"    {label:<32}  N/A")
            else:
                bar = "█" * int(v * 26)
                print(f"    {label:<32}  {v:.4f}  {bar}")

    print(f"\n  {'─' * (W - 2)}")
    print("  OVERALL SCORES")
    print("  " + "─" * (W - 2))
    for k in ALL_EVAL_KEYS:
        v = metrics["overall"].get(k)
        label = eval_labels.get(k, k)
        if v is None:
            print(f"    {label:<32}  N/A")
        else:
            bar = "█" * int(v * 26)
            print(f"    {label:<32}  {v:.4f}  {bar}")

    cs = metrics.get("composite_score")
    print(f"\n  {'─' * (W - 2)}")
    print(f"  COMPOSITE SYSTEM SCORE                    "
          f"{'N/A' if cs is None else f'{cs:.4f}'}")
    print("═" * W + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Clinical System Evaluation Pipeline v2")
    log.info("Project   : %s", PROJECT_NAME)
    log.info("Mock mode : %s", SKIP_REAL_PIPELINE)
    log.info("Examples  : %d", len(EVAL_DATASET))
    log.info("=" * 60)

    # ── Register stop signal handlers ─────────────────────────────────────────
    # Must be done in the main thread (Python restriction).
    signal.signal(signal.SIGINT,  _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)
    log.info("Signal handlers registered — press Ctrl+C to stop cleanly.")

    # ── Validate required env vars ────────────────────────────────────────────
    required = ["OPENAI_API_KEY", "LANGSMITH_API_KEY"]
    missing  = [v for v in required if not os.getenv(v)]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {missing}\n"
            "Fill them in your .env file and re-run."
        )

    # Warn loudly if judge base URL is missing — the most common cause of 401
    if not os.getenv("EVAL_JUDGE_BASE_URL"):
        log.warning(
            "⚠️  EVAL_JUDGE_BASE_URL is not set. "
            "If you are using an llmod.ai key the judge LLM will return 401. "
            "Add EVAL_JUDGE_BASE_URL=https://api.llmod.ai/v1 to your .env"
        )

    ls_client = LangSmithClient(
        api_url=os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
        api_key=os.getenv("LANGSMITH_API_KEY"),
    )

    if SKIP_REAL_PIPELINE:
        system = None
        log.info("MOCK MODE active — real pipeline will not be called.")
    else:
        log.info("Initialising ManagerAgent …")
        from backend.main import initialize, execute_pipeline

        class _SystemWrapper:
            def run(self, s): return execute_pipeline(s)

        initialize()
        system = _SystemWrapper()
        log.info("ManagerAgent ready.")

    example_lookup = {f"EVAL-{ex['id']}": ex for ex in EVAL_DATASET}
    dataset_id     = create_or_update_langsmith_dataset(ls_client)
    log.info("Dataset id=%s", dataset_id)

    target = make_pipeline_target(system, example_lookup)

    log.info("Launching evaluate() …")
    eval_results = evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[
            RoutingAccuracyEvaluator(),
            SafetyRobustnessEvaluator(),
            ClinicalToneEvaluator(),
            ToolSelectionEvaluator(),
            ToolSequencingEvaluator(),
            TaskCompletenessEvaluator(),
        ],
        experiment_prefix="clinical-eval",
        metadata={
            "eval_version":  "v2",
            "mock_mode":     str(SKIP_REAL_PIPELINE),
            "dataset_size":  len(EVAL_DATASET),
            "run_timestamp": datetime.now().isoformat(),
            "evaluators":    ", ".join(ALL_EVAL_KEYS),
        },
        max_concurrency=1,
    )

    results_list = list(eval_results)
    metrics      = aggregate_results(results_list)

    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"eval_metrics_{ts}.json"
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Metrics saved → %s", output_path)

    print_report(metrics)
    print(f"📊 Full traces: {os.getenv('LANGSMITH_ENDPOINT', 'https://smith.langchain.com')}"
          f"  |  Project: {PROJECT_NAME}\n")

    return metrics


if __name__ == "__main__":
    main()