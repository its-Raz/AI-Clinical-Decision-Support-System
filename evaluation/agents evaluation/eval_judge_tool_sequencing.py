"""
eval_judge_tool_sequencing.py — Tool Sequencing Evaluator.

Answers the question:
  "Did the system execute its tools and nodes in the correct order,
   respecting data dependencies and the defined workflow?"

The expected execution sequence for this LangGraph pipeline is:

  [1] manager_node     → calls judge_decision → writes request_type, next_step
       ↓
  [2] specialist_node  → reads request_type, patient_id, lab_result/image_path
       ↓                  writes lab_insights / vision_insights / evidence_insights
  [3] deliver_node     → reads specialist insights → writes final_report

A sequencing failure occurs when:
  - deliver_node fires before specialist_node produces its output
  - specialist_node runs without manager_node having set request_type
  - final_report is present but specialist output fields are empty
    (deliver ran but had no insights to synthesise — silent failure)
  - judge_decision appears AFTER specialist output (manager ran late)

Scoring:
  1.0 → Correct sequence confirmed; all data dependencies respected
  0.5 → Sequence mostly correct but a recoverable dependency gap exists
         (e.g., deliver got empty insights but still produced a valid report)
  0.0 → Sequence violated (specialist ran before manager, deliver ran
         before specialist, or critical output field missing at wrong stage)

Applied to: ALL examples (A, B, C) — sequence must hold regardless of content.
"""

import logging

from langsmith.evaluation import RunEvaluator, EvaluationResult
from langsmith.schemas import Run, Example

from eval_judge_base import get_judge_llm, call_judge, extract_all_traces

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM = """You are a workflow auditor evaluating whether an AI agent pipeline executed
its steps in the correct order, respecting all data dependencies.
Return ONLY a JSON object — no preamble, no explanation outside the JSON."""

_USER = """
## Task
Verify that the clinical AI pipeline executed its nodes in the correct sequence.

## Expected Pipeline Sequence
Step 1 → manager_node:    calls judge_decision tool, writes `request_type` and `next_step`
Step 2 → specialist_node: reads `request_type`, executes analysis, writes specialist insights
Step 3 → deliver_node:    reads specialist insights, writes `final_report`

For non-clinical/adversarial requests, specialist_node is SKIPPED:
Step 1 → manager_node  →  Step 2 → deliver_node (directly)

## Observed Final State
- request_type populated:           {request_type_set}
- next_step populated:              {next_step_set}
- lab_insights populated:           {lab_insights_set}
- vision_insights populated:        {vision_insights_set}
- evidence_insights populated:      {evidence_insights_set}
- final_report populated:           {final_report_set}

## Requires Specialist (from dataset)
{requires_specialist}

## Full Message Trace (in chronological order — earlier messages appear first)
{full_trace}

## Sequencing Rules
1. A [Judge] trace message MUST appear before any specialist output in the message list.
2. If requires_specialist=True, at least one specialist insight field must be populated
   BEFORE final_report was generated.
3. If requires_specialist=False, specialist insight fields must all be empty.
4. final_report must be non-empty for every completed run.
5. The [Manager → deliver_node] trace must appear AFTER any specialist trace messages.

## Scoring
- 1.0 → All 5 sequencing rules satisfied. No dependency violations.
- 0.5 → Minor sequence gap: specialist ran but produced empty output, yet deliver
         still completed (graceful degradation — acceptable but suboptimal).
- 0.0 → Rules 1, 3, or 5 violated (critical out-of-order execution), OR
         final_report is empty (pipeline did not complete).

## Output — return ONLY this JSON:
{{
  "score": <float: 0.0, 0.5, or 1.0>,
  "sequence_valid": <bool>,
  "rule_violations": ["<list violated rule numbers as strings, e.g. '2', '5'>"],
  "dependency_gap_detected": <bool>,
  "reason": "<one sentence explaining your verdict>"
}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class ToolSequencingEvaluator(RunEvaluator):
    """
    Verifies the pipeline executed nodes in the correct order.

    Why sequence matters in LangGraph:
      LangGraph's StateGraph guarantees topological order at compile time,
      but runtime failures (e.g., a specialist returning None) can cause
      deliver_node to synthesise from empty inputs — a silent quality failure
      that routing accuracy alone would not catch. This evaluator surfaces it.
    """

    def __init__(self):
        self.llm = get_judge_llm()

    def evaluate_run(self, run: Run, example: Example = None) -> EvaluationResult:
        inputs  = run.inputs  or {}
        outputs = run.outputs or {}
        ref     = (example.outputs or {}) if example else {}

        requires_specialist = ref.get("requires_specialist", False)
        messages            = outputs.get("messages", [])
        full_trace          = extract_all_traces(messages, max_chars=1500)

        # Snapshot of which state fields got populated
        state_snapshot = {
            "request_type_set":      bool(outputs.get("request_type")),
            "next_step_set":         bool(outputs.get("next_step")),
            "lab_insights_set":      bool(outputs.get("lab_insights")),
            "vision_insights_set":   bool(outputs.get("vision_insights")),
            "evidence_insights_set": bool(outputs.get("evidence_insights")),
            "final_report_set":      bool(outputs.get("final_report")),
        }

        result = call_judge(
            self.llm, _SYSTEM,
            _USER.format(
                requires_specialist=requires_specialist,
                full_trace=full_trace,
                **state_snapshot,
            ),
        )

        score             = float(result.get("score", 0.0))
        sequence_valid    = result.get("sequence_valid", False)
        rule_violations   = result.get("rule_violations", [])
        dependency_gap    = result.get("dependency_gap_detected", False)
        reason            = result.get("reason", "Parse failure.")

        log.info(
            "ToolSequencing [%s]: score=%.1f valid=%s violations=%s",
            inputs.get("patient_id", "?"), score, sequence_valid, rule_violations,
        )

        return EvaluationResult(
            key="tool_sequencing",
            score=score,
            comment=(
                f"Sequence valid: {sequence_valid} | "
                f"Rule violations: {rule_violations or 'none'} | "
                f"Dependency gap: {dependency_gap} | "
                f"State: {state_snapshot} | "
                f"{reason}"
            ),
        )
