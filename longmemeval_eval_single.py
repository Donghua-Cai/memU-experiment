"""Evaluate a single LongMemEval sample using existing memory."""
import json
import os
import sys
from pathlib import Path
import dotenv

dotenv.load_dotenv()

from response_agent import ResponseAgent
from evaluate_agent import EvaluateAgent
from memu.utils import setup_logging

logger = setup_logging(__name__, enable_flush=True)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a single LongMemEval sample using existing memory")
    parser.add_argument(
        "--sample-file",
        required=True,
        help="Path to a single LongMemEval JSON sample (e.g., data/00001_7161e7e2.json)",
    )
    parser.add_argument(
        "--memory-dir",
        default="memory_longmem_eval",
        help="Base memory directory used in longmemeval_test.py",
    )
    parser.add_argument(
        "--chat-deployment",
        default="gpt-4o-mini",
        help="Azure OpenAI chat deployment",
    )
    parser.add_argument(
        "--use-profile",
        type=str,
        default="none",
        help="Use profile to answer: none/search/prompt",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run comprehensive evaluation (slower)",
    )

    args = parser.parse_args()

    sample_path = Path(args.sample_file)
    if not sample_path.exists():
        raise FileNotFoundError(sample_path)

    with open(sample_path, "r", encoding="utf-8") as f:
        sample = json.load(f)

    question_id = sample.get("question_id")
    if not question_id:
        raise ValueError("Missing question_id in sample")

    question = sample.get("question", "")
    standard_answer = sample.get("answer", "")

    # Memory dir layout: memory_longmem_eval/longmemeval/<question_id>
    memory_dir = Path(args.memory_dir) / str(question_id)
    if not memory_dir.exists():
        raise FileNotFoundError(f"Memory dir not found: {memory_dir}")

    # Characters are named user###/assistant### in longmemeval_test.py
    # Derive index from filename prefix if possible (00001_...)
    sample_index = None
    stem = sample_path.stem
    if "_" in stem and stem.split("_")[0].isdigit():
        sample_index = int(stem.split("_")[0])

    if sample_index is None:
        # Fallback: try sample_index=1
        sample_index = 1

    user_name = f"user{sample_index:03d}"
    assistant_name = f"assistant{sample_index:03d}"
    characters = [user_name, assistant_name]

    response_agent = ResponseAgent(
        chat_deployment=args.chat_deployment,
        memory_dir=str(memory_dir),
    )

    evaluate_agent = EvaluateAgent(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
        chat_deployment="gpt-4o-mini",
    )

    logger.info(f"Evaluating question_id={question_id} using memory at {memory_dir}")
    logger.info(f"Characters: {characters}")

    answer_result = response_agent.execute_tool(
        "answer_question",
        question=question,
        characters=characters,
        use_profile=args.use_profile,
    )

    if not answer_result.get("success", False):
        raise RuntimeError(answer_result.get("error", "Failed to answer question"))

    generated_answer = answer_result.get("answer", "")

    evaluation = evaluate_agent.evaluate_answer_accuracy(
        question, generated_answer, standard_answer
    )

    output = {
        "question_id": question_id,
        "question": question,
        "standard_answer": standard_answer,
        "generated_answer": generated_answer,
        "evaluation": evaluation,
    }

    if args.analyze:
        output["comprehensive_evaluation"] = evaluate_agent.comprehensive_evaluation(
            question=question,
            generated_answer=generated_answer,
            standard_answer=standard_answer,
            retrieved_events=answer_result.get("retrieved_events", []),
        )

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
