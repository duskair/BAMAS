"""
Main Experiment: Full Pipeline Inference with Trained High-Level Policy (Parallel Version)
Evaluates the trained system on MATH test dataset across different budgets using multiprocessing.
"""
import torch
import random
import os
import json
import argparse
import sys
import numpy as np
from tqdm import tqdm
import time
from datetime import datetime
import torch.multiprocessing as mp
import queue
import threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from eapae_agent_sys.utils.config_loader import ConfigLoader
from eapae_agent_sys.data_processing.math_loader import load_math_dataset
from eapae_agent_sys.planning.semantic_filter import SemanticFilter
from eapae_agent_sys.planning.low_level_instantiator import LowLevelInstantiator
from eapae_agent_sys.execution.execution_engine import ExecutionEngine
from eapae_agent_sys.planning.high_level_policy import HighLevelPolicy
from eapae_agent_sys.utils.llm_api import LLM_API
from eapae_agent_sys.utils.evaluation import evaluate_math_success

# Difficulty-driven model-tier selection: The library has two tiers: 'Low' (llama-3.1-8b, cheap) and 'High' (llama-3.3-70b, strong). 
# Easy tasks are restricted to the cheap tier; hard tasks to the strong tier; medium tasks may use either and the ILP picks under the budget.
DIFFICULTY_BANDS = [
    (0.4, ["Low"]),           # easy   -> cheap tier only
    (0.6, ["Low", "High"]),   # medium -> either tier, ILP decides under budget
    (1.01, ["High"]),          # hard   -> strong tier only
]

def map_difficulty_to_tier(difficulty_score: float) -> list:
    """Map a difficulty score in [0, 1] to the allowed agent config_levels."""
    for threshold, tiers in DIFFICULTY_BANDS:
        if difficulty_score < threshold:
            return tiers
    return DIFFICULTY_BANDS[-1][1]

def monitor_workers(workers: list, heartbeat_dict: dict, stop_event: threading.Event, timeout: int):
    """
    A thread that monitors worker processes and terminates them if they hang.
    """
    while not stop_event.wait(20.0):
        for p in workers:
            if not p.is_alive():
                continue
            last_beat = heartbeat_dict.get(p.pid)
            if last_beat is None:
                continue
            if time.time() - last_beat > timeout:
                p.terminate()
def worker(work_queue, result_queue, heartbeat_dict, model_state_dict, device_type, difficulty=True):
    """Worker process to run a single experiment.

    Args:
        difficulty: If True (default), difficulty drives the *model tier* the candidate agent pool is restricted to the config_levels for 
        this task's difficulty via map_difficulty_to_tier (easy -> cheap 8b only), so the ILP can only build a team from that tier. 
        If False, the difficulty score is still recorded, but has no effect.
    """
    worker_pid = os.getpid()
    heartbeat_dict[worker_pid] = time.time()
    if device_type == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    config_loader = ConfigLoader(
        agent_library_file="0_agent_library_math.yml",
        training_params_file="3_training_params_math.yml"
    )
    params = config_loader.params
    llm_api = LLM_API()
    semantic_filter = SemanticFilter(config_loader, llm_api)
    low_level_instantiator = LowLevelInstantiator(config_loader)
    execution_engine = ExecutionEngine(config_loader, low_level_instantiator)
    embedding_dim = 384
    num_patterns = len(config_loader.patterns['patterns'])
    max_budget = params.get('training', {}).get('budget_range', [0, 2000])[1]
    policy_network = HighLevelPolicy(
        embedding_dim=embedding_dim,
        num_patterns=num_patterns,
        max_budget=max_budget,
        device=device
    ).to(device)
    if model_state_dict is not None:
        policy_network.load_state_dict(model_state_dict)
    policy_network.eval()
    all_patterns = config_loader.patterns['patterns']
    while True:
        work_item = work_queue.get()
        if work_item is None:
            break
        heartbeat_dict[worker_pid] = time.time()
        work_id, task, budget, experiment_metadata = work_item
        try:
            with torch.no_grad():
                task_description = [task['problem']]
                budget_tensor = torch.tensor([budget], device=device, dtype=torch.float32)
                selected_action, log_prob, logits = policy_network(
                    task_description, budget_tensor, is_deterministic=True
                )
                pattern_idx = selected_action.item()
                selected_pattern = all_patterns[pattern_idx]
        except Exception as e:
            print(f"Policy inference failed: {e}")
            pattern_idx = random.randint(0, len(all_patterns) - 1)
            selected_pattern = all_patterns[pattern_idx]
        is_correct = False
        actual_cost = 0.0
        planning_feasible = False
        inference_successful = False
        final_context = None
        difficulty_score = None
        try:
            candidate_agents = semantic_filter.filter_candidates(task['problem'], collaboration_pattern=selected_pattern)
            if not candidate_agents:
                final_context = {"history": ["Planning failed: Semantic filter returned no candidate agents."]}
            else:
                with torch.no_grad():
                    predictor = execution_engine.get_difficulty_predictor("math")
                    difficulty_score = predictor([task['problem']], apply_sigmoid=True).item()

                if difficulty:
                    allowed_tiers = map_difficulty_to_tier(difficulty_score)
                    filtered = [a for a in candidate_agents
                                if config_loader.agents.get(a, {}).get('config_level') in allowed_tiers]
                    if filtered:
                        candidate_agents = filtered
                        print(f"\n[*] MATH Difficulty->Tier: difficulty {difficulty_score:.4f} -> tiers {allowed_tiers} ({len(filtered)} candidate agents)")
                    else:
                        print(f"\n[*] MATH Difficulty->Tier: no candidates in tiers {allowed_tiers}; keeping full pool")
                else:
                    print(f"\n[*] MATH Difficulty recorded (baseline, no effect) -> Difficulty: {difficulty_score:.4f}")

                agent_pool, is_feasible = low_level_instantiator.solve(
                    pattern_or_name=selected_pattern,
                    budget=budget,
                    candidate_agent_ids=candidate_agents
                )
                planning_feasible = is_feasible
                if is_feasible:
                    final_context, actual_cost = execution_engine.execute_hybrid(
                        agent_pool, budget, task['problem'],
                        collaboration_pattern=selected_pattern,
                        dataset_type="math"
                    )
                    success_code = evaluate_math_success(final_context, task['solution'])
                    is_correct = success_code == 1
                    inference_successful = True
                else:
                    final_context = {"history": ["Planning failed: No feasible agent team could be formed for the given budget and pattern."]}
        except Exception as e:
            print(f"[Worker-{worker_pid}] Inference failed: {e}")
            actual_cost = budget
            is_correct = False
            final_context = {"history": [f"Execution failed: {str(e)}"]}
        result = {
            "work_id": work_id,
            "task_id": task.get('idx', 'unknown'),
            "task_description": task['problem'],
            "ground_truth_answer": task['solution'],
            "difficulty_score": difficulty_score,
            "difficulty": difficulty,
            "budget": budget,
            "action": pattern_idx,
            "pattern_name": selected_pattern['name'],
            "is_correct": is_correct,
            "actual_cost": actual_cost,
            "planning_feasible": planning_feasible,
            "inference_successful": inference_successful,
            "timestamp": datetime.now().isoformat(),
            **experiment_metadata
        }
        result_queue.put(result)
def main():
    parser = argparse.ArgumentParser(description="Main experiment: Parallel full pipeline inference on MATH test set")
    parser.add_argument("--model_path", type=str, help="Path to trained high-level policy model")
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of test samples to use (-1 for all)")
    parser.add_argument("--num_budget_steps", type=int, default=3, help="Number of budget levels (default: 3 for MATH)")
    parser.add_argument("--beginwith", type=int, default=0, help="Task index to start from (0-based)")
    parser.add_argument("--output_dir", type=str, default="experiments/results/math", help="Output directory for results")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of parallel worker processes")
    parser.add_argument("--diff_off", action="store_true",
                        help="Baseline: do not use difficulty to restrict model tier (record difficulty but ignore it)")
    args = parser.parse_args()
    difficulty = not args.diff_off
    print("="*80)
    print("MAIN EXPERIMENT: PARALLEL FULL PIPELINE INFERENCE (MATH DATASET)")
    print(f"Difficulty -> tier: {'ON' if difficulty else 'OFF (baseline)'}")
    print("="*80)
    mp.set_start_method("spawn", force=True)
    config_loader = ConfigLoader(
        agent_library_file="0_agent_library_math.yml",
        training_params_file="3_training_params_math.yml"
    )
    params = config_loader.params
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = params.get('training', {}).get('seed', 42)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    test_dataset = load_math_dataset("data/math", split="sampled_test")
    if args.beginwith > 0:
        test_dataset = test_dataset[args.beginwith:]
    if args.num_samples != -1:
        test_dataset = test_dataset[:args.num_samples]
    model_state_dict = None
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = os.path.join(params['outputs']['checkpoints_dir'], "high_level_policy_offline_best_math.pth")
    if os.path.exists(model_path):
        print(f"Loading trained MATH model from: {model_path}")
        model_state_dict = torch.load(model_path, map_location='cpu')
    else:
        print(f"Warning: MATH model not found at {model_path}")
    budget_range = params.get('training', {}).get('budget_range', [500, 2000])
    budgets = np.linspace(budget_range[0], budget_range[1], args.num_budget_steps).tolist()
    print(f"Testing with budgets: {budgets}")
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manager = mp.Manager()
    heartbeat_dict = manager.dict()
    work_queue = mp.Queue()
    result_queue = mp.Queue()
    work_items = []
    work_items_map = {}
    for task_idx, task in enumerate(test_dataset):
        for budget_idx, budget in enumerate(budgets):
            work_id = f"exp_{task_idx + args.beginwith}_budget_{budget_idx}"
            experiment_metadata = {
                "experiment_id": f"exp_{timestamp}",
                "task_idx": task_idx + args.beginwith,
                "budget_idx": budget_idx,
                "total_budgets": len(budgets),
                "total_tasks": len(test_dataset)
            }
            work_item = (work_id, task, budget, experiment_metadata)
            work_items.append(work_item)
            work_items_map[work_id] = work_item
    for item in work_items:
        work_queue.put(item)
    for _ in range(args.num_workers):
        work_queue.put(None)
    workers = []
    device_type = str(device).split(':')[0]
    for _ in range(args.num_workers):
        p = mp.Process(target=worker, args=(work_queue, result_queue, heartbeat_dict, model_state_dict, device_type, difficulty))
        p.start()
        workers.append(p)
    TIMEOUT_SECONDS = 300
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_workers,
        args=(workers, heartbeat_dict, stop_event, TIMEOUT_SECONDS),
        daemon=True
    )
    monitor_thread.start()
    all_results = []
    total_work_items = len(work_items)
    try:
        with tqdm(total=total_work_items, desc="Running Experiments") as pbar:
            while pbar.n < total_work_items:
                try:
                    result = result_queue.get(timeout=5)
                    all_results.append(result)
                    pbar.set_postfix({
                        "Pattern": result['action'],
                        "Success": result['is_correct'],
                        "Budget": f"{result['budget']:.0f}"
                    })
                    pbar.update(1)
                except queue.Empty:
                    if not any(p.is_alive() for p in workers):
                        break
    finally:
        print("\n--- Initiating Cleanup ---")
        successful_ids = {result.get('work_id') for result in all_results if result.get('work_id')}
        all_ids = set(work_items_map.keys())
        failed_ids = all_ids - successful_ids
        if failed_ids:
            print(f"[Cleanup] Found {len(failed_ids)} incomplete tasks. Adding them as failure records.")
            for failed_id in sorted(list(failed_ids)):
                work_id, task, budget, experiment_metadata = work_items_map[failed_id]
                failure_result = {
                    "work_id": work_id,
                    "task_id": task.get('idx', 'unknown'),
                    "task_description": task['problem'],
                    "ground_truth_answer": task['solution'],
                    "difficulty": difficulty,
                    "budget": budget,
                    "action": 0,
                    "pattern_name": "unknown",
                    "is_correct": False,
                    "actual_cost": budget,
                    "planning_feasible": False,
                    "inference_successful": False,
                    "timestamp": datetime.now().isoformat(),
                    "reason": "worker_timeout_or_failure",
                    **experiment_metadata
                }
                all_results.append(failure_result)
        stop_event.set()
        monitor_thread.join(timeout=5)
        for p in workers:
            if p.is_alive():
                p.terminate()
        for p in workers:
            p.join()
    mode = "diffon" if difficulty else "diffoff"
    results_file = os.path.join(args.output_dir, f"experiment_results_{mode}_{timestamp}.jsonl")
    print(f"\nSaving detailed results to: {results_file}")
    with open(results_file, 'w', encoding='utf-8') as f:
        for result in all_results:
            f.write(json.dumps(result) + "\n")
    with open(results_file, 'r', encoding='utf-8') as f:
        all_results = [json.loads(line) for line in f if line.strip()]
    summary_file = os.path.join(args.output_dir, f"experiment_summary_{mode}_{timestamp}.json")
    print(f"Generating summary statistics: {summary_file}")
    all_patterns = config_loader.patterns['patterns']
    stats_by_budget = {}
    stats_by_pattern = {}
    stats_by_budget_pattern = {}
    for budget in budgets:
        budget_results = [r for r in all_results if r['budget'] == budget]
        total = len(budget_results)
        correct = sum(1 for r in budget_results if r['is_correct'])
        feasible = sum(1 for r in budget_results if r['planning_feasible'])
        avg_cost = float(np.mean([r['actual_cost'] for r in budget_results])) if total > 0 else 0.0
        stats_by_budget[f"budget_{budget:.0f}"] = {
            "total_samples": total,
            "accuracy": correct / total if total > 0 else 0,
            "planning_feasible_rate": feasible / total if total > 0 else 0,
            "average_cost": avg_cost
        }
        budget_pattern_stats = {}
        for pattern_idx in range(len(all_patterns)):
            pattern_results = [r for r in budget_results if r['action'] == pattern_idx]
            total_pattern = len(pattern_results)
            correct_pattern = sum(1 for r in pattern_results if r['is_correct'])
            feasible_pattern = sum(1 for r in pattern_results if r['planning_feasible'])
            avg_pattern_cost = float(np.mean([r['actual_cost'] for r in pattern_results])) if total_pattern > 0 else 0.0
            budget_pattern_stats[f"pattern_{pattern_idx}"] = {
                "pattern_name": all_patterns[pattern_idx]['name'],
                "total_samples": total_pattern,
                "accuracy": correct_pattern / total_pattern if total_pattern > 0 else 0,
                "planning_feasible_rate": feasible_pattern / total_pattern if total_pattern > 0 else 0,
                "selection_frequency": total_pattern / total if total > 0 else 0,
                "average_cost": avg_pattern_cost
            }
        stats_by_budget_pattern[f"budget_{budget:.0f}"] = budget_pattern_stats
    for pattern_idx in range(len(all_patterns)):
        pattern_results = [r for r in all_results if r['action'] == pattern_idx]
        total = len(pattern_results)
        correct = sum(1 for r in pattern_results if r['is_correct'])
        feasible = sum(1 for r in pattern_results if r['planning_feasible'])
        avg_cost = float(np.mean([r['actual_cost'] for r in pattern_results])) if total > 0 else 0.0
        stats_by_pattern[f"pattern_{pattern_idx}"] = {
            "pattern_name": all_patterns[pattern_idx]['name'],
            "total_samples": total,
            "accuracy": correct / total if total > 0 else 0,
            "planning_feasible_rate": feasible / total if total > 0 else 0,
            "selection_frequency": total / len(all_results) if len(all_results) > 0 else 0,
            "average_cost": avg_cost
        }
    total_results = len(all_results)
    overall_accuracy = sum(1 for r in all_results if r['is_correct']) / total_results if total_results > 0 else 0
    overall_feasible_rate = sum(1 for r in all_results if r['planning_feasible']) / total_results if total_results > 0 else 0
    summary = {
        "experiment_info": {
            "timestamp": timestamp,
            "model_path": model_path,
            "difficulty": difficulty,
            "total_tasks": len(test_dataset),
            "budgets_tested": budgets,
            "total_experiments": total_results,
            "num_workers": args.num_workers
        },
        "overall_performance": {
            "accuracy": overall_accuracy,
            "planning_feasible_rate": overall_feasible_rate
        },
        "performance_by_budget": stats_by_budget,
        "performance_by_pattern": stats_by_pattern,
        "performance_by_budget_and_pattern": stats_by_budget_pattern
    }
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"Total experiments: {total_results}")
    print(f"Overall accuracy: {overall_accuracy:.3f} ({overall_accuracy*100:.1f}%)")
    print(f"Planning feasible rate: {overall_feasible_rate:.3f} ({overall_feasible_rate*100:.1f}%)")
    print(f"\nPerformance by Budget:")
    for budget_key, stats in stats_by_budget.items():
        budget_val = budget_key.replace('budget_', '')
        print(f"  Budget {budget_val}: {stats['accuracy']:.3f} accuracy, {stats['planning_feasible_rate']:.3f} feasible")
    print(f"\nPattern Results by Budget:")
    for budget_key, pattern_stats in stats_by_budget_pattern.items():
        budget_val = budget_key.replace('budget_', '')
        print(f"  Budget {budget_val}:")
        for pattern_key, stats in pattern_stats.items():
            print(f"    - {stats['pattern_name']}: {stats['accuracy']:.3f} accuracy, {stats['planning_feasible_rate']:.3f} feasible ({stats['selection_frequency']*100:.1f}% of budget samples)")
    print(f"\nPattern Summary:")
    for pattern_key, stats in stats_by_pattern.items():
        print(f"  {stats['pattern_name']}: selection {stats['selection_frequency']*100:.1f}%, avg cost {stats['average_cost']:.2f}")
    print(f"\nResults saved to:")
    print(f"  Detailed: {results_file}")
    print(f"  Summary: {summary_file}")
    print("="*80)
if __name__ == "__main__":
    main() 