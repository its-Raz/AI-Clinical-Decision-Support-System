"""
eval_judge_tone.py — Clinical Tone & Synthesis Quality Evaluator.

Answers the question:
  "Did the Delivery Node successfully transform the specialist's raw clinical
   insights into a professional, empathetic, and patient-friendly report?"

Uses a five-dimension rubric so failures can be traced back to a specific
layer of the pipeline (specialist output quality vs delivery prompt quality):
  1. Empathy & Tone       — warm, human, non-alarmist
  2. Medical Accuracy     — reflects findings correctly, appropriate caveats
  3. Clarity              — jargon translated, next steps actionable
  4. Completeness         — all significant findings addressed
  5. Structure            — well-organised, appropriate length

Final score = average of 5 dimensions, normalised to 0.0-1.0.

Applied to: Category A (Core Functional) examples only.
"""

import logging

from langsmith.evaluation import RunEvaluator, EvaluationResult
from langsmith.schemas import Run, Example

from eval_judge_base import get_judge_llm, call_judge, make_na_result

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM = """You are a clinical communications expert evaluating patient-facing medical messages.
You assess whether the message is empathetic, accurate, structured, and genuinely useful to a patient.
Return ONLY a JSON object — no preamble, no explanation outside the JSON."""

_USER = """
## Task
Score the quality of a patient-facing clinical summary produced by a medical AI delivery node.

## Request Type
{request_type}

## Raw Clinical Insights (what the specialist agent produced internally)
---
{raw_insights}
---

## Final Patient-Facing Message (what the patient actually sees)
---
{final_report}
---

## Evaluation Rubric — score each dimension 1-10:

1. **Empathy & Tone** (1-10)
   - Is the message warm, reassuring, and human?
   - Does it avoid being cold, robotic, or alarmist?

2. **Medical Accuracy** (1-10)
   - Does it accurately reflect the specialist's findings?
   - Does it avoid introducing incorrect medical claims?
   - Does it appropriately caveat that this is not a diagnosis?

3. **Clarity & Accessibility** (1-10)
   - Is medical jargon translated into plain language?
   - Are actionable next steps clearly stated?

4. **Completeness** (1-10)
   - Are all significant findings from the specialist output covered?
   - Are abnormal markers specifically called out?

5. **Structure** (1-10)
   - Is the message well-organised (greeting → findings → next steps)?
   - Is it appropriately concise (not too long, not too short)?

## Scoring Instructions
- Compute the AVERAGE of the 5 dimension scores.
- Return final_score = average / 10 (range: 0.0 to 1.0).

## Output — return ONLY this JSON:
{{
  "empathy_score":      <int 1-10>,
  "accuracy_score":     <int 1-10>,
  "clarity_score":      <int 1-10>,
  "completeness_score": <int 1-10>,
  "structure_score":    <int 1-10>,
  "final_score":        <float 0.0-1.0>,
  "key_strengths":      "<one sentence on what the message did well>",
  "key_weaknesses":     "<one sentence on the main area for improvement>"
}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class ClinicalToneEvaluator(RunEvaluator):
    """
    Evaluates the Delivery Node's patient-facing final_report quality.

    Why five sub-dimensions rather than one score:
      A single "quality" score is too opaque for debugging. If accuracy is
      high but clarity is low, that points to the DELIVERY_PROMPT needing
      simplification. If completeness is low, the specialist agent may be
      truncating its output. Each dimension maps to a fixable component.
    """

    def __init__(self):
        self.llm = get_judge_llm()

    def evaluate_run(self, run: Run, example: Example = None) -> EvaluationResult:
        inputs  = run.inputs  or {}
        outputs = run.outputs or {}
        ref     = (example.outputs or {}) if example else {}

        category = ref.get("category", "")
        if category != "A":
            return make_na_result(
                "clinical_tone",
                "clinical tone only evaluated for Category A (core functional).",
            )

        request_type = outputs.get("request_type", "unknown")
        final_report = (outputs.get("final_report") or "")

        if not final_report:
            return EvaluationResult(
                key="clinical_tone",
                score=0.0,
                comment="Final report was empty — delivery node failed completely.",
            )

        # Select the correct specialist insight field for this request type
        if request_type == "blood_test_analysis":
            raw_insights = outputs.get("lab_insights") or "Not available."
        elif request_type == "image_lesion_analysis":
            raw_insights = outputs.get("vision_insights") or "Not available."
        else:
            raw_insights = outputs.get("evidence_insights") or "Not available."

        result = call_judge(
            self.llm, _SYSTEM,
            _USER.format(
                request_type=request_type,
                raw_insights=str(raw_insights)[:1200],
                final_report=final_report[:2000],
            ),
        )

        final_score = float(result.get("final_score", 0.0))
        strengths   = result.get("key_strengths", "N/A")
        weaknesses  = result.get("key_weaknesses", "N/A")

        sub = {k: result.get(k) for k in [
            "empathy_score", "accuracy_score", "clarity_score",
            "completeness_score", "structure_score",
        ]}

        log.info(
            "ClinicalTone [%s / %s]: final_score=%.2f sub=%s",
            inputs.get("patient_id", "?"), request_type, final_score, sub,
        )

        return EvaluationResult(
            key="clinical_tone",
            score=final_score,
            comment=(
                f"Empathy={sub['empathy_score']} | "
                f"Accuracy={sub['accuracy_score']} | "
                f"Clarity={sub['clarity_score']} | "
                f"Completeness={sub['completeness_score']} | "
                f"Structure={sub['structure_score']} | "
                f"Strengths: {strengths} | "
                f"Weaknesses: {weaknesses}"
            ),
        )
