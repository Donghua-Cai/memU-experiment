"""
LongMemEval Reconstruct Test with MemU Memory Modules

Differences from longmemeval_test.py:
- Input is a directory of JSON samples (longmemeval_s_items_reconstruct)
- haystack_dates are ignored; per-turn time is read from each message's "time"
- orig_session_id is ignored
"""

import json
import os
import sys
import ast
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import dotenv

dotenv.load_dotenv()

from mem_agent import MemAgent
from response_agent import ResponseAgent
from evaluate_agent import EvaluateAgent
from memu.utils import setup_logging

logger = setup_logging(__name__, enable_flush=True)

args_global = None


class LongMemEvalReconstructTester:
    def __init__(
        self,
        azure_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        chat_deployment: str = "gpt-4o-mini",
        use_entra_id: bool = False,
        api_version: str = "2024-02-01",
        memory_dir: str = "memory_longmem_eval",
        max_workers: int = 3,
        per_sample_memory: bool = True,
    ):
        self.mem_agent = MemAgent(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            chat_deployment=chat_deployment,
            use_entra_id=use_entra_id,
            api_version=api_version,
            memory_dir=memory_dir,
        )

        self.response_agent = ResponseAgent(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            chat_deployment=chat_deployment,
            use_entra_id=use_entra_id,
            api_version=api_version,
            memory_dir=memory_dir,
        )

        eval_azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT") or ""
        eval_api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY") or ""

        self.evaluate_agent = EvaluateAgent(
            azure_endpoint=eval_azure_endpoint,
            api_key=eval_api_key,
            chat_deployment="gpt-4o-mini",
            use_entra_id=use_entra_id,
            api_version=api_version,
        )

        self.max_workers = max_workers
        self.per_sample_memory = per_sample_memory
        self.base_memory_dir = Path(memory_dir)

        self.results = []
        self.processing_time = 0.0

        self.log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.error_log_file = f"longmemeval_reconstruct_error_log_{self.log_timestamp}.txt"
        self._init_error_log()

        logger.info(
            "LongMemEval Reconstruct Tester initialized with MemAgent (memory) and ResponseAgent (QA)"
        )
        logger.info(f"QA error log file: {self.error_log_file}")

    def _init_error_log(self):
        try:
            with open(self.error_log_file, "w", encoding="utf-8") as f:
                f.write(
                    f"LongMemEval Reconstruct Error Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                f.write("=" * 80 + "\n")
                f.write("This log contains detailed information for incorrect answers.\n\n")
        except Exception as e:
            logger.error(f"Failed to initialize error log file: {e}")

    def _log_qa_error(
        self,
        question_id: str,
        question: str,
        generated_answer: str,
        standard_answer: str,
        question_type: str,
        retrieved_content: str = "",
        explanation: str = "",
        evaluation_details: Optional[Dict] = None,
    ):
        try:
            content_lines = []
            content_lines.append(f"\n{'=' * 80}")
            content_lines.append(f"QUESTION ID: {question_id}")
            content_lines.append(f"QUESTION TYPE: {question_type}")
            content_lines.append(f"TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            content_lines.append(f"{'=' * 80}\n")

            content_lines.append(f"QUESTION:\n{question}\n")
            content_lines.append(
                f"RETRIEVED CONTENT (From Memory):\n{retrieved_content if retrieved_content else 'No content retrieved'}\n"
            )
            content_lines.append(f"GENERATED ANSWER:\n{generated_answer}\n")
            content_lines.append(f"STANDARD ANSWER:\n{standard_answer}\n")
            content_lines.append(f"EVALUATION EXPLANATION:\n{explanation}\n")

            if evaluation_details:
                content_lines.append(f"{'=' * 60}")
                content_lines.append("COMPREHENSIVE EVALUATION RESULTS")
                content_lines.append(f"{'=' * 60}\n")
                content_lines.append(json.dumps(evaluation_details, indent=2, ensure_ascii=False))
                content_lines.append("\n")

            content_lines.append(f"{'=' * 80}\n")

            error_content = "\n".join(content_lines)
            with open(self.error_log_file, "a", encoding="utf-8") as f:
                f.write(error_content)

        except Exception as e:
            logger.error(f"Failed to write to error log: {e}")

    def _set_sample_memory_dir(self, sample_id: str):
        if not self.per_sample_memory:
            return

        sample_dir = self.base_memory_dir / "longmemeval_reconstruct" / str(sample_id)
        sample_dir.mkdir(parents=True, exist_ok=True)

        self.mem_agent.memory_dir = sample_dir
        self.response_agent.memory_dir = sample_dir
        self.response_agent.clear_embedding_cache()

    def _extract_sessions(self, sample: Dict, role_map: Dict[str, str]) -> List[Tuple[int, List[Dict], str]]:
        sessions = []
        haystack_sessions = sample.get("haystack_sessions", [])

        for i, session in enumerate(haystack_sessions):
            utterances = []
            session_date = "Unknown Date"
            for msg in session:
                if isinstance(msg, dict):
                    role = msg.get("role", "unknown")
                    speaker = role_map.get(role, role)
                    content = msg.get("content", "")
                    turn_time = msg.get("time", "")
                    if turn_time and session_date == "Unknown Date":
                        session_date = turn_time
                    if turn_time:
                        text = f"[{turn_time}] {content}"
                    else:
                        text = content
                    utterances.append({"speaker": speaker, "text": text})
            sessions.append((i, utterances, session_date))

        return sessions

    def _get_characters_for_sample(self, sample_index: int) -> Tuple[List[str], Dict[str, str]]:
        user_name = f"user{sample_index:03d}"
        assistant_name = f"assistant{sample_index:03d}"
        role_map = {"user": user_name, "assistant": assistant_name}
        return [user_name, assistant_name], role_map

    def _process_single_session(
        self, session_data: Tuple[int, List[Dict], str], characters: List[str]
    ) -> Dict:
        session_idx, session_utterances, session_date = session_data

        try:
            logger.info(
                f"Processing session {session_idx} with {len(session_utterances)} utterances on {session_date}"
            )

            _use_image = getattr(args_global, "use_image", False)
            update_result = self.mem_agent.update_character_memory(
                session_data=session_utterances,
                session_date=session_date,
                characters=characters,
                use_image=_use_image,
            )

            if update_result.get("success", False):
                return {
                    "session_idx": session_idx,
                    "success": True,
                    "utterances_count": len(session_utterances),
                    "session_date": session_date,
                }
            else:
                error_msg = update_result.get("error", "Unknown error")
                return {
                    "session_idx": session_idx,
                    "success": False,
                    "error": error_msg,
                    "utterances_count": len(session_utterances),
                    "session_date": session_date,
                }

        except Exception as e:
            logger.error(f"Exception processing session {session_idx}: {e}")
            return {
                "session_idx": session_idx,
                "success": False,
                "error": str(e),
                "utterances_count": len(session_utterances) if session_utterances else 0,
                "session_date": session_date,
            }

    def _process_sessions_parallel(
        self,
        sessions: List[Tuple[int, List[Dict], str]],
        characters: List[str],
        max_workers: int = 3,
    ) -> List[Dict]:
        if not sessions:
            return []

        session_results = []
        completed_count = 0
        total_sessions = len(sessions)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_session = {
                executor.submit(self._process_single_session, session, characters): session[0]
                for session in sessions
            }

            for future in as_completed(future_to_session):
                session_idx = future_to_session[future]
                completed_count += 1

                try:
                    result = future.result()
                    session_results.append(result)
                    status = "\u2713" if result["success"] else "\u2717"
                    logger.info(
                        f"[{completed_count}/{total_sessions}] {status} session {session_idx} completed"
                    )
                except Exception as e:
                    logger.error(
                        f"[{completed_count}/{total_sessions}] \u2717 session {session_idx} generated exception: {e}"
                    )
                    session_results.append(
                        {
                            "session_idx": session_idx,
                            "success": False,
                            "error": str(e),
                            "utterances_count": 0,
                            "session_date": "Unknown",
                        }
                    )

        for character_name in characters:
            self.mem_agent.clean_profile(character_name)

        session_results.sort(key=lambda x: x["session_idx"])
        return session_results

    def _build_retrieved_content_summary(self, answer_result: Dict) -> str:
        context_used = answer_result.get("context_used", {})
        retrieved_contents = []

        characters_searched = context_used.get("characters_searched", [])
        search_keywords = context_used.get("search_keywords", [])

        final_content = answer_result.get("retrieved_events", [])
        iteration_log = answer_result.get("iteration_log", [])
        total_events_found = context_used.get("total_events_found", 0)
        content_pieces = context_used.get("content_pieces", 0)

        if total_events_found > 0:
            retrieved_contents.append(f"Relevant Events Found: {total_events_found} events")
            if search_keywords:
                retrieved_contents.append(f"Search Keywords: {', '.join(search_keywords)}")

        if characters_searched:
            retrieved_contents.append(f"Characters Searched: {', '.join(characters_searched)}")

        if content_pieces > 0:
            retrieved_contents.append(f"Content Pieces Retrieved: {content_pieces}")

        if final_content:
            retrieved_contents.append("\n--- ACTUAL RETRIEVED CONTENT ---")
            for i, content in enumerate(final_content[:20], 1):
                if isinstance(content, dict):
                    content_text = content.get("text", content.get("event", str(content)))
                    retrieved_contents.append(f"\n{i}. {content_text}")
                    string_score = content.get("string_score", 0.0)
                    bm25_score = content.get("bm25_score", 0.0)
                    semantic_score = content.get("semantic_score", 0.0)
                    combined_score = content.get("combined_score", 0.0)
                    retrieved_contents.append(
                        f"    Scores - String: {string_score:.3f}, BM25: {bm25_score:.3f}, Semantic: {semantic_score:.3f}, Combined: {combined_score:.3f}"
                    )
                else:
                    retrieved_contents.append(f"\n{i}. {content}")

        if iteration_log:
            retrieved_contents.append("\n--- RETRIEVAL ITERATIONS ---")
            for i, iteration in enumerate(iteration_log, 1):
                if isinstance(iteration, dict):
                    iteration_summary = iteration.get("summary", str(iteration))
                    retrieved_contents.append(f"\nIteration {i}: {iteration_summary}")
                else:
                    retrieved_contents.append(f"\nIteration {i}: {iteration}")

        return "\n".join(retrieved_contents) if retrieved_contents else "ResponseAgent provided direct answer"

    def _evaluate_answer(self, question: str, generated_answer: str, standard_answer: str) -> Dict:
        try:
            result = self.evaluate_agent.evaluate_answer_accuracy(
                question, generated_answer, standard_answer
            )
            if result["success"]:
                return {
                    "is_correct": result["is_correct"],
                    "explanation": result["explanation"],
                    "evaluation_text": result["evaluation_text"],
                }
            return {
                "is_correct": False,
                "explanation": f"Evaluation failed: {result.get('error', 'Unknown error')}",
                "evaluation_text": "",
            }
        except Exception as e:
            logger.error(f"Failed to evaluate answer: {e}")
            return {
                "is_correct": False,
                "explanation": f"Evaluation failed: {e}",
                "evaluation_text": "",
            }

    def process_sample(self, sample: Dict, sample_index: int) -> Dict:
        start_time = time.time()

        try:
            question_id = sample.get("question_id", f"sample_{sample_index}")
            question_type = sample.get("question_type", "unknown")
            question = sample.get("question", "")
            standard_answer = sample.get("answer", "")

            self._set_sample_memory_dir(question_id)

            characters, role_map = self._get_characters_for_sample(sample_index)
            sessions = self._extract_sessions(sample, role_map)

            if getattr(args_global, "force_resum", False) or self.per_sample_memory:
                self.mem_agent.clear_character_memory(characters)

            session_results = self._process_sessions_parallel(
                sessions, characters, max_workers=self.max_workers
            )

            if not getattr(args_global, "enable_response", True):
                generated_answer = ""
                evaluation = {"is_correct": False, "explanation": "Response disabled"}
                retrieved_content = ""
                retrieved_events = []
            elif getattr(args_global, "no_eval", False):
                generated_answer = ""
                evaluation = {"is_correct": False, "explanation": "Evaluation skipped"}
                retrieved_content = ""
                retrieved_events = []
            else:
                use_profile = getattr(args_global, "use_profile", "none")
                answer_result = self.response_agent.execute_tool(
                    "answer_question",
                    question=question,
                    characters=characters,
                    use_profile=use_profile,
                )

                if answer_result.get("success", False):
                    generated_answer = answer_result.get("answer", "No answer generated")
                    retrieved_content = self._build_retrieved_content_summary(answer_result)
                    retrieved_events = answer_result.get("retrieved_events", [])
                else:
                    error_msg = answer_result.get("error", "Failed to generate answer")
                    generated_answer = f"Error: {error_msg}"
                    retrieved_content = f"Error occurred during answer generation: {error_msg}"
                    retrieved_events = []

                evaluation = self._evaluate_answer(question, generated_answer, standard_answer)

                analyze_on = getattr(args_global, "analyze_on", "wrong")
                if analyze_on == "all" or (
                    analyze_on == "wrong" and not evaluation["is_correct"]
                ):
                    comprehensive_evaluation = None
                    try:
                        comprehensive_evaluation = self.evaluate_agent.comprehensive_evaluation(
                            question=question,
                            generated_answer=generated_answer,
                            standard_answer=standard_answer,
                            retrieved_events=retrieved_events,
                        )
                    except Exception as e:
                        logger.error(f"Comprehensive evaluation failed: {e}")
                        comprehensive_evaluation = None

                    self._log_qa_error(
                        question_id=question_id,
                        question=question,
                        generated_answer=generated_answer,
                        standard_answer=standard_answer,
                        question_type=question_type,
                        retrieved_content=retrieved_content,
                        explanation=evaluation.get("explanation", ""),
                        evaluation_details=comprehensive_evaluation,
                    )

            processing_time = time.time() - start_time

            return {
                "question_id": question_id,
                "question_type": question_type,
                "question": question,
                "standard_answer": standard_answer,
                "generated_answer": generated_answer,
                "is_correct": evaluation.get("is_correct", False),
                "explanation": evaluation.get("explanation", ""),
                "retrieved_content": retrieved_content,
                "sessions_total": len(sessions),
                "sessions_processed": len(session_results),
                "processing_time": processing_time,
                "success": True,
            }

        except Exception as e:
            logger.error(f"Failed to process sample: {e}")
            return {
                "question_id": sample.get("question_id", f"sample_{sample_index}"),
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time,
            }

    def run_test(self, data_dir: str, sample_use: Optional[str] = None) -> Dict:
        start_time = time.time()

        try:
            data_path = Path(data_dir)
            if not data_path.exists():
                raise FileNotFoundError(data_dir)

            files = sorted([p for p in data_path.glob("*.json") if p.is_file()])
            data = []
            for p in files:
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data.append(json.load(f))
                except Exception as e:
                    logger.error(f"Failed to read {p}: {e}")

            if sample_use:
                try:
                    parsed_value = ast.literal_eval(sample_use)
                    if isinstance(parsed_value, int):
                        if parsed_value <= 0:
                            raise ValueError("sample_use number must be positive")
                        sample_count = min(parsed_value, len(data))
                        data = data[:sample_count]
                        logger.info(
                            f"Using first {sample_count} samples (indices 0-{sample_count-1})"
                        )
                    elif isinstance(parsed_value, list):
                        sample_indices = parsed_value
                        valid_indices = [
                            i for i in sample_indices if isinstance(i, int) and 0 <= i < len(data)
                        ]
                        data = [data[i] for i in valid_indices]
                        logger.info(f"Using samples at specific indices: {valid_indices}")
                        if len(valid_indices) < len(sample_indices):
                            invalid_indices = [
                                i for i in sample_indices if i not in valid_indices
                            ]
                            logger.warning(
                                f"Ignored invalid/out-of-range indices: {invalid_indices}"
                            )
                    else:
                        raise ValueError("sample_use must be either an integer or a list of integers")
                except Exception as e:
                    logger.error(f"Failed to parse sample_use '{sample_use}': {e}")
                    logger.info("Using all samples instead")
            else:
                logger.info("No sample_use specified, using all samples")

            logger.info(f"Starting test with {len(data)} samples")

            all_results = []
            total_questions = 0
            total_correct = 0
            total_processed = 0

            for i, sample in enumerate(data, 1):
                logger.info(f"\n=== Processing Sample {i}/{len(data)} ===")
                result = self.process_sample(sample, i)
                all_results.append(result)

                if result.get("success"):
                    total_questions += 1
                    total_processed += 1
                    if result.get("is_correct"):
                        total_correct += 1

                logger.info(f"Sample {i} completed in {result.get('processing_time', 0):.2f}s")

            total_time = time.time() - start_time
            overall_accuracy = total_correct / total_processed if total_processed > 0 else 0.0

            summary = {
                "total_samples": len(data),
                "successful_samples": sum(1 for r in all_results if r.get("success")),
                "total_questions": total_questions,
                "total_correct": total_correct,
                "overall_accuracy": overall_accuracy,
                "total_time": total_time,
                "avg_time_per_sample": total_time / len(data) if data else 0.0,
            }

            self.results = all_results
            self.processing_time = total_time

            return {
                "success": True,
                "summary": summary,
                "detailed_results": all_results,
            }

        except Exception as e:
            logger.error(f"Test run failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "summary": {},
                "detailed_results": [],
            }

    def print_results(self):
        if not self.results:
            print("No results to display")
            return

        total_samples = len(self.results)
        successful_samples = sum(1 for r in self.results if r.get("success"))
        total_questions = sum(1 for r in self.results if r.get("success"))
        total_correct = sum(1 for r in self.results if r.get("is_correct"))

        overall_accuracy = total_correct / total_questions if total_questions > 0 else 0.0

        print("\n" + "=" * 60)
        print("LONGMEMEVAL RECONSTRUCT TEST RESULTS - MEMU")
        print("=" * 60)
        print(f"Samples processed: {successful_samples}/{total_samples}")
        print(f"Total questions: {total_questions}")
        print(f"Total correct: {total_correct}")
        print(f"Overall accuracy: {overall_accuracy:.2%}")
        print(f"Total processing time: {self.processing_time:.2f}s")
        print(f"Average time per sample: {self.processing_time / total_samples:.2f}s")
        print("=" * 60)

        incorrect = total_questions - total_correct
        if incorrect > 0:
            print("ERROR LOG INFORMATION")
            print("=" * 60)
            print(f"Total incorrect answers: {incorrect}")
            print(f"Detailed error information saved to: {self.error_log_file}")
        else:
            print("All questions answered correctly. No error log entries generated.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="LongMemEval reconstruct test with MemU")
    parser.add_argument(
        "--data-dir",
        default="data/longmemeval_s_items_reconstruct",
        help="Path to LongMemEval reconstruct items directory",
    )
    parser.add_argument(
        "--sample-use",
        type=str,
        default="5",
        help='Sample indices to use. Can be a single number (e.g., "10") or a list (e.g., "[0, 1, 3]").',
    )
    parser.add_argument(
        "--memory-dir",
        default="memory_longmem_eval",
        help="Directory for memory files",
    )
    parser.add_argument(
        "--chat-deployment", default="gpt-4o-mini", help="Azure OpenAI chat deployment"
    )
    parser.add_argument(
        "--max-workers", type=int, default=3, help="Maximum number of parallel workers"
    )
    parser.add_argument(
        "--use-image",
        type=lambda x: x.lower() != "false",
        default=True,
        help="Insert image caption to conversation (default: True)",
    )
    parser.add_argument(
        "--use-profile", type=str, default="none", help="Use the profile to answer the questions"
    )
    parser.add_argument(
        "--force-resum", action="store_true", help="Force to redo the memory summarization"
    )
    parser.add_argument("--no-eval", action="store_true", help="Do not evaluate the results")
    parser.add_argument(
        "--analyze-on", type=str, default="wrong", help='Do detailed analysis on "all", "wrong", or "none"'
    )
    parser.add_argument(
        "--no-per-sample-memory",
        action="store_false",
        dest="per_sample_memory",
        default=True,
        help="Disable per-sample memory isolation (default: True)",
    )
    parser.add_argument(
        "--enable-response",
        action="store_true",
        default=True,
        help="Enable ResponseAgent (default: True). Use --disable-response to turn off.",
    )
    parser.add_argument(
        "--disable-response",
        action="store_false",
        dest="enable_response",
        help="Disable ResponseAgent (memory-only run)",
    )

    args = parser.parse_args()

    global args_global
    args_global = args

    tester = LongMemEvalReconstructTester(
        memory_dir=args.memory_dir,
        chat_deployment=args.chat_deployment,
        max_workers=args.max_workers,
        per_sample_memory=args.per_sample_memory,
    )

    args_dict = vars(args)
    script_command = " ".join(sys.argv)

    results = {"args": args_dict, "script": script_command}

    logger.info("Starting LongMemEval Reconstruct Test with MemU")
    results |= tester.run_test(args.data_dir, args.sample_use)

    if results["success"]:
        tester.print_results()
        output_file = f"longmemeval_reconstruct_test_results_{tester.log_timestamp}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Detailed results saved to: {output_file}")
    else:
        logger.error(f"Test failed: {results.get('error', 'Unknown error')}")
        logger.info(f"Check error log for any partial results: {tester.error_log_file}")


if __name__ == "__main__":
    main()
