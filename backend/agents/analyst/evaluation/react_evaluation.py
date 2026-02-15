"""
ReAct Node Evaluation Script

Tests the ReAct agent's performance on known patient cases.

Evaluation Criteria:
1. Tool Usage: Did it call the right tools?
2. Efficiency: How many iterations/tools?
3. Completeness: Did it gather all necessary info?
4. Correctness: Does the summary match expected diagnosis?
"""
import sys
import os
from typing import Dict, List, Any
from datetime import datetime
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agents.analyst.nodes.react import react_node
TEST_CASES = [

    {
        "name": "P002 - Prediabetes",
        "patient_id": "P002",
        "patient_info": {"age": 58, "sex": "M"},
        "lab_result": {
            "test_name": "Glucose",
            "value": 115,
            "unit": "mg/dL",
            "flag": "high"
        },
        "expected_tools": ["get_patient_history", "check_reference_range", "search_medical_knowledge"],
        "expected_keywords": ["prediabetes", "glucose", "lifestyle", "rising"],
        "expected_severity": "abnormal",
        "min_tools": 3,
        "max_iterations": 5
    }

]


# ==================== Evaluation Functions ====================

def evaluate_tool_usage(result: Dict, expected_tools: List[str]) -> Dict[str, Any]:
    """
    Evaluate if the agent called the right tools.

    Returns:
        dict with:
        - tools_called: list of tool names
        - expected_tools: list of expected tools
        - missing_tools: tools that should have been called but weren't
        - extra_tools: tools called but not expected
        - score: 0-1 (percentage of expected tools called)
    """
    tools_called = list(result.get("observations", {}).keys())

    missing_tools = [t for t in expected_tools if t not in tools_called]
    extra_tools = [t for t in tools_called if t not in expected_tools]

    score = len([t for t in expected_tools if t in tools_called]) / len(expected_tools)

    return {
        "tools_called": tools_called,
        "expected_tools": expected_tools,
        "missing_tools": missing_tools,
        "extra_tools": extra_tools,
        "score": score,
        "passed": len(missing_tools) == 0
    }


def evaluate_efficiency(result: Dict, max_iterations: int, min_tools: int) -> Dict[str, Any]:
    """
    Evaluate if the agent was efficient.

    Returns:
        dict with:
        - iterations_used: number of iterations
        - max_iterations: maximum allowed
        - tools_used: number of tools called
        - min_tools_required: minimum required
        - efficiency_score: 0-1
    """
    iterations = result.get("react_iterations", 0)
    tool_count = len(result.get("tool_calls", []))

    # Penalize if too many iterations or too few tools
    iteration_score = 1.0 if iterations <= max_iterations else 0.5
    tool_score = 1.0 if tool_count >= min_tools else 0.5

    efficiency_score = (iteration_score + tool_score) / 2

    return {
        "iterations_used": iterations,
        "max_iterations": max_iterations,
        "tools_used": tool_count,
        "min_tools_required": min_tools,
        "efficiency_score": efficiency_score,
        "passed": iterations <= max_iterations and tool_count >= min_tools
    }


def evaluate_completeness(result: Dict) -> Dict[str, Any]:
    """
    Evaluate if all necessary information was gathered.

    Checks:
    - Patient history retrieved
    - Reference range checked
    - Medical knowledge searched
    """
    observations = result.get("observations", {})

    has_patient_history = "get_patient_history" in observations
    has_reference_check = "check_reference_range" in observations
    has_medical_knowledge = "search_medical_knowledge" in observations

    completeness_score = sum([
        has_patient_history,
        has_reference_check,
        has_medical_knowledge
    ]) / 3.0

    return {
        "has_patient_history": has_patient_history,
        "has_reference_check": has_reference_check,
        "has_medical_knowledge": has_medical_knowledge,
        "completeness_score": completeness_score,
        "passed": completeness_score == 1.0
    }


def evaluate_output_quality(result: Dict, expected_keywords: List[str]) -> Dict[str, Any]:
    """
    Evaluate the quality of the agent's summary.

    Checks:
    - Contains expected keywords
    - Summary is not empty
    - Summary is not too short
    """
    summary = result.get("react_summary", "").lower()

    # Check for expected keywords
    keywords_found = [kw for kw in expected_keywords if kw.lower() in summary]
    keyword_score = len(keywords_found) / len(expected_keywords) if expected_keywords else 0

    # Check length
    has_summary = len(summary) > 0
    is_substantial = len(summary) > 50  # At least 50 characters

    quality_score = (keyword_score + has_summary + is_substantial) / 3.0

    return {
        "summary_length": len(summary),
        "expected_keywords": expected_keywords,
        "keywords_found": keywords_found,
        "keyword_score": keyword_score,
        "quality_score": quality_score,
        "passed": keyword_score >= 0.5 and is_substantial
    }


def evaluate_test_case(test_case: Dict) -> Dict[str, Any]:
    """
    Run a single test case and evaluate results.
    """
    print("\n" + "=" * 70)
    print(f"TEST: {test_case['name']}")
    print("=" * 70)

    # Prepare state
    state = {
        "patient_id": test_case["patient_id"],
        "patient_info": test_case["patient_info"],
        "lab_result": test_case["lab_result"]
    }

    # Run ReAct node
    try:
        result = react_node(state)

        # Check for errors
        if "react_error" in result:
            return {
                "test_case": test_case["name"],
                "status": "ERROR",
                "error": result["react_error"],
                "overall_score": 0.0
            }

        # Evaluate
        tool_usage_eval = evaluate_tool_usage(result, test_case["expected_tools"])
        efficiency_eval = evaluate_efficiency(result, test_case["max_iterations"], test_case["min_tools"])
        completeness_eval = evaluate_completeness(result)
        quality_eval = evaluate_output_quality(result, test_case["expected_keywords"])

        # Calculate overall score
        overall_score = (
                tool_usage_eval["score"] * 0.3 +
                efficiency_eval["efficiency_score"] * 0.2 +
                completeness_eval["completeness_score"] * 0.3 +
                quality_eval["quality_score"] * 0.2
        )

        # Print results
        print(f"\n📊 EVALUATION RESULTS:")
        print(f"\n1. Tool Usage: {tool_usage_eval['score']:.2f}")
        print(f"   Called: {tool_usage_eval['tools_called']}")
        if tool_usage_eval['missing_tools']:
            print(f"   ❌ Missing: {tool_usage_eval['missing_tools']}")
        else:
            print(f"   ✅ All expected tools called")

        print(f"\n2. Efficiency: {efficiency_eval['efficiency_score']:.2f}")
        print(f"   Iterations: {efficiency_eval['iterations_used']}/{efficiency_eval['max_iterations']}")
        print(f"   Tools used: {efficiency_eval['tools_used']} (min: {efficiency_eval['min_tools_required']})")

        print(f"\n3. Completeness: {completeness_eval['completeness_score']:.2f}")
        print(f"   Patient history: {'✅' if completeness_eval['has_patient_history'] else '❌'}")
        print(f"   Reference check: {'✅' if completeness_eval['has_reference_check'] else '❌'}")
        print(f"   Medical knowledge: {'✅' if completeness_eval['has_medical_knowledge'] else '❌'}")

        print(f"\n4. Output Quality: {quality_eval['quality_score']:.2f}")
        print(f"   Summary length: {quality_eval['summary_length']} chars")
        print(f"   Keywords found: {quality_eval['keywords_found']}")
        if len(quality_eval['keywords_found']) < len(quality_eval['expected_keywords']):
            missing_kw = [kw for kw in quality_eval['expected_keywords'] if kw not in quality_eval['keywords_found']]
            print(f"   Missing keywords: {missing_kw}")

        print(f"\n🎯 OVERALL SCORE: {overall_score:.2f} / 1.00")
        print(
            f"   Status: {'✅ PASS' if overall_score >= 0.7 else '⚠️ NEEDS IMPROVEMENT' if overall_score >= 0.5 else '❌ FAIL'}")

        return {
            "test_case": test_case["name"],
            "status": "PASS" if overall_score >= 0.7 else "FAIL",
            "overall_score": overall_score,
            "tool_usage": tool_usage_eval,
            "efficiency": efficiency_eval,
            "completeness": completeness_eval,
            "quality": quality_eval,
            "summary": result.get("react_summary", "")
        }

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            "test_case": test_case["name"],
            "status": "ERROR",
            "error": str(e),
            "overall_score": 0.0
        }


# ==================== Main ====================

def run_evaluation():
    """
    Run all test cases and generate report.
    """
    print("\n" + "=" * 70)
    print("REACT NODE EVALUATION")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Test cases: {len(TEST_CASES)}")

    results = []

    for test_case in TEST_CASES:
        result = evaluate_test_case(test_case)
        results.append(result)

    # Generate summary
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_tests = len(results)
    passed_tests = len([r for r in results if r["status"] == "PASS"])
    failed_tests = len([r for r in results if r["status"] == "FAIL"])
    error_tests = len([r for r in results if r["status"] == "ERROR"])

    avg_score = sum([r["overall_score"] for r in results]) / total_tests if total_tests > 0 else 0

    print(f"\nTotal Tests: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {failed_tests}")
    print(f"⚠️  Errors: {error_tests}")
    print(f"\n📊 Average Score: {avg_score:.2f} / 1.00")

    # Pass rate
    pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    print(f"📈 Pass Rate: {pass_rate:.1f}%")

    # Save results to JSON
    output_file = f"react_eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {output_file}")

    return results


if __name__ == "__main__":
    run_evaluation()