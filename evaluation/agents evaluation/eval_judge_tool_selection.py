"""
eval_judge_tool_selection.py — Tool Selection Accuracy Evaluator.

Answers the question:
  "Did the system identify and invoke the CORRECT tool(s) for the given task?"

In this clinical multi-agent system, "tools" operate at two levels:

  LEVEL 1 — Manager/Judge tools:
    - `judge_decision` tool: must be called by the Manager for every request.
      Its `accepted_category` arg determines which specialist is activated.

  LEVEL 2 — Specialist-internal tools:
    - blood_test_analyst  : lab analysis functions (flagging, trend detection)
    - evidence_analyst    : RAG retrieval tool against Pinecone/MedlinePlus
    - skin_care_analyst   : YOLOv11 bounding-box inference

  The evaluator verifies:
    1. Was `judge_decision` called with the correct category?
    2. Was the appropriate specialist activated (or correctly skipped)?
    3. For adversarial/OOD requests: were specialist tools correctly NOT invoked?

Scoring:
  1.0 → All correct tools selected; no inappropriate tools invoked
  0.5 → Correct specialist selected but sub-optimal tool usage within it
         (e.g., judge_decision called with wrong category but system recovered)
  0.0 → Wrong tool selected, specialist invoked for adversarial input, or
         judge_decision not called at all

Applied to: ALL examples (A, B, C)
"""

import logging

from langsmith.evaluation import RunEvaluator, EvaluationResult
from langsmith.schemas import Run, Example

from eval_judge_base import get_judge_llm, call_judge, extract_judge_trace, extract_all_traces

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM = """You are a technical auditor evaluating an AI agent system's tool selection decisions.
You determine whether the correct tools were chosen for each task.
Return ONLY a JSON object — no preamble, no explanation outside the JSON."""

_USER = """
## Task
Evaluate whether the clinical AI system selected the correct tools for this request.

## Patient Input
{user_input}

## Request Category
{subcategory}

## Expected Tool Usage
- Expected specialist activated: "{expected_route}"
- Should a specialist tool have been called at all: {requires_specialist}

## Actual System Behaviour (from internal trace)
Judge trace:
{judge_trace}

Full message trace:
{full_trace}

## State of Specialist Output Fields
- lab_insights populated:      {lab_populated}
- vision_insights populated:   {vision_populated}
- evidence_insights populated: {evidence_populated}

## Tool Inventory for this System
| Tool / Agent            | When it SHOULD be used                                      |
|-------------------------|-------------------------------------------------------------|
| judge_decision          | ALWAYS — Manager must call this to classify every request   |
| blood_test_analyst      | When patient asks about lab results, blood values, metrics  |
| skin_care_analyst       | When patient uploads a skin/lesion image                    |
| evidence_analyst        | When patient asks a general clinical/medical question       |
| deliver (no specialist) | When request is adversarial, OOD, spam, or unsupported      |

## Scoring Criteria
- 1.0 → judge_decision was called with the correct category AND the correct
         specialist (or lack thereof) was activated. No inappropriate tools invoked.
- 0.5 → Correct specialist ultimately activated, but the judge_decision category
         argument was imprecise, OR a minor unnecessary tool call was made.
- 0.0 → Wrong specialist activated, OR specialist tool invoked for an adversarial/OOD
         request, OR judge_decision not called at all (no [Judge] trace present).

## Output — return ONLY this JSON:
{{
  "score": <float: 0.0, 0.5, or 1.0>,
  "judge_decision_called": <bool>,
  "judge_category_correct": <bool>,
  "specialist_correctly_invoked": <bool>,
  "inappropriate_tool_called": <bool>,
  "reason": "<one sentence explaining your verdict>"
}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class ToolSelectionEvaluator(RunEvaluator):
    """
    Verifies the system chose the right tools for the given task.

    Key insight: in an agentic system, tool selection failures compound —
    invoking the wrong specialist wastes tokens, may return irrelevant
    clinical content, and (for adversarial inputs) is a safety violation.
    We detect this by inspecting both the Judge trace and which specialist
    output fields were populated in the final state.
    """

    def __init__(self):
        self.llm = get_judge_llm()

    def evaluate_run(self, run: Run, example: Example = None) -> EvaluationResult:
        inputs  = run.inputs  or {}
        outputs = run.outputs or {}
        ref     = (example.outputs or {}) if example else {}

        user_input          = inputs.get("raw_user_input", "")
        expected_route      = ref.get("expected_route", "unknown")
        requires_specialist = ref.get("requires_specialist", False)
        subcategory         = ref.get("subcategory", "unknown")
        messages            = outputs.get("messages", [])

        judge_trace = extract_judge_trace(messages)
        full_trace  = extract_all_traces(messages, max_chars=1200)

        # Inspect which specialist output fields were populated
        lab_populated     = bool(outputs.get("lab_insights"))
        vision_populated  = bool(outputs.get("vision_insights"))
        evidence_populated = bool(outputs.get("evidence_insights"))

        result = call_judge(
            self.llm, _SYSTEM,
            _USER.format(
                user_input=user_input,
                subcategory=subcategory,
                expected_route=expected_route,
                requires_specialist=requires_specialist,
                judge_trace=judge_trace,
                full_trace=full_trace,
                lab_populated=lab_populated,
                vision_populated=vision_populated,
                evidence_populated=evidence_populated,
            ),
        )

        score                      = float(result.get("score", 0.0))
        judge_called               = result.get("judge_decision_called", False)
        specialist_correctly_invoked = result.get("specialist_correctly_invoked", False)
        inappropriate_tool         = result.get("inappropriate_tool_called", False)
        reason                     = result.get("reason", "Parse failure.")

        log.info(
            "ToolSelection [%s]: expected=%s score=%.1f judge_called=%s specialist_ok=%s",
            inputs.get("patient_id", "?"), expected_route, score,
            judge_called, specialist_correctly_invoked,
        )

        return EvaluationResult(
            key="tool_selection",
            score=score,
            comment=(
                f"Expected: '{expected_route}' | "
                f"judge_decision called: {judge_called} | "
                f"Specialist correctly invoked: {specialist_correctly_invoked} | "
                f"Inappropriate tool: {inappropriate_tool} | "
                f"{reason}"
            ),
        )
