"""Compare a vLLM run directly with the frozen E251 candidate workload."""
import hashlib
import json
from pathlib import Path

from scoring_v3 import answer, correct


VLLM_PATH = Path('results/vllm_fair_e251_order_20261102.jsonl')
HF_ROOT = Path('results/e251/abba_rerun_20261102/trial1_candidate')
OUTPUT = Path('results/vllm_fair_e251_order_20261102.audit.json')


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def task_map(rows):
    return {row['id']: row for row in rows if row.get('kind') == 'task'}


def score(tasks):
    failed = sorted(task_id for task_id, task in tasks.items()
                    if not correct(answer(task['rounds'][-1]['text']), task['expected']))
    return len(tasks) - len(failed), failed


def main():
    vllm_rows = jsonl(VLLM_PATH)
    vllm_meta = next(row for row in vllm_rows if row.get('kind') == 'metadata')
    vllm_summary = next(row for row in vllm_rows if row.get('kind') == 'summary')
    vllm_tasks = task_map(vllm_rows)
    hf_manifest = json.loads((HF_ROOT / 'gate.json').read_text())
    hf_rows = []
    for batch in hf_manifest['batches']:
        hf_rows.extend(json.loads(Path(batch['output']).read_text())['tasks'])
    hf_tasks = {row['id']: row for row in hf_rows}
    expected_order = [row['id'] for row in hf_rows]
    vllm_order = vllm_meta.get('task_order', [])
    hf_score, hf_failed = score(hf_tasks)
    vllm_score, vllm_failed = score(vllm_tasks)
    token_diffs = []
    tool_diffs = []
    for task_id in expected_order:
        left = hf_tasks[task_id]
        right = vllm_tasks.get(task_id)
        if right is None:
            token_diffs.append(task_id)
            continue
        left_rounds, right_rounds = left.get('rounds', []), right.get('rounds', [])
        if [turn.get('tokens') for turn in left_rounds] != [turn.get('tokens') for turn in right_rounds]:
            token_diffs.append(task_id)
        left_tools = [turn.get('tool_output') for turn in left_rounds if 'code' in turn]
        right_tools = [turn.get('tool_output') for turn in right_rounds if 'code' in turn]
        if left_tools != right_tools:
            tool_diffs.append(task_id)
    token_count = sum(len(turn.get('tokens', [])) for task in vllm_tasks.values()
                      for turn in task.get('rounds', []))
    tool_calls = sum('code' in turn for task in vllm_tasks.values()
                     for turn in task.get('rounds', []))
    generation = [row for row in vllm_rows if row.get('kind') == 'generate_batch']
    batches = [row for row in vllm_rows if row.get('kind') == 'tool_batch']
    first_turn_batches = [row for row in generation if row.get('turn') == 0]
    checks = {
        'task_order_matches_hf': vllm_order == expected_order,
        'all_128_unique_tasks': len(vllm_tasks) == 128 and set(vllm_tasks) == set(hf_tasks),
        'same_b16_first_turn_layout': len(first_turn_batches) == 8 and all(row.get('requests') == 16 for row in first_turn_batches),
        '128_tool_calls': tool_calls == 128,
        'all_tasks_two_rounds_one_tool': all(len(task.get('rounds', [])) == 2 and
                                             sum('code' in turn for turn in task.get('rounds', [])) == 1
                                             for task in vllm_tasks.values()),
        'same_v3_quality_as_hf': vllm_score >= hf_score and set(vllm_failed) <= set(hf_failed),
        'same_data': vllm_meta.get('data_sha256') == hashlib.sha256(Path('data/gsm8k_128.jsonl').read_bytes()).hexdigest(),
        'same_batch_size_and_tool_workers': vllm_meta.get('batch_size') == 16 and vllm_meta.get('tool_workers') == 1,
    }
    candidate_manifests = [json.loads((HF_ROOT.parent / name / 'gate.json').read_text())
                           for name in ('trial1_candidate', 'trial2_candidate')]
    hf_candidate_mean = sum(row['makespan_seconds'] for row in candidate_manifests) / 2
    hf_load_mean = sum(row['load_seconds'] for row in candidate_manifests) / 2
    report = {
        'verified_comparable_workload': checks['task_order_matches_hf'] and checks['all_128_unique_tasks']
            and checks['same_b16_first_turn_layout'] and checks['same_data']
            and checks['same_batch_size_and_tool_workers'],
        'quality_passed': checks['all_tasks_two_rounds_one_tool'] and checks['same_v3_quality_as_hf'],
        'checks': checks,
        'vllm': {
            'framework': vllm_meta['framework'], 'version': vllm_meta['vllm_version'],
            'workflow_seconds': vllm_summary['makespan_seconds'],
            'load_and_engine_init_seconds': vllm_meta['load_seconds'],
            'tasks': len(vllm_tasks), 'tokens': token_count, 'tool_calls': tool_calls,
            'score_v3': vllm_score, 'failed_ids_v3': vllm_failed,
            'token_trajectory_diff_ids_vs_hf': token_diffs,
            'tool_output_diff_ids_vs_hf': tool_diffs,
            'tool_batch_records': len(batches),
            'second_turn_request_counts': [row.get('requests') for row in generation if row.get('turn') == 1],
            'raw_sha256': hashlib.sha256(VLLM_PATH.read_bytes()).hexdigest(),
        },
        'hf_e251_candidate': {
            'workflow_seconds': hf_manifest['makespan_seconds'],
            'candidate_abba_mean_seconds': hf_candidate_mean,
            'load_seconds': hf_manifest['load_seconds'],
            'candidate_load_mean_seconds': hf_load_mean,
            'tasks': len(hf_tasks),
            'tokens': sum(len(turn.get('tokens', [])) for task in hf_tasks.values()
                          for turn in task.get('rounds', [])),
            'tool_calls': sum('code' in turn for task in hf_tasks.values()
                              for turn in task.get('rounds', [])),
            'score_v3': hf_score, 'failed_ids_v3': hf_failed,
        },
        'comparison': {
            'vllm_minus_hf_candidate_abba_mean_seconds': vllm_summary['makespan_seconds'] - hf_candidate_mean,
            'vllm_percent_slower_vs_hf_candidate_abba_mean': 100 * (vllm_summary['makespan_seconds'] / hf_candidate_mean - 1),
            'cold_total_vllm_seconds': vllm_summary['makespan_seconds'] + vllm_meta['load_seconds'],
            'cold_total_hf_candidate_abba_mean_seconds': hf_candidate_mean + hf_load_mean,
        },
        'memory_note': 'NPU HBM peak was not sampled during this single run; no memory comparison is claimed.',
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
