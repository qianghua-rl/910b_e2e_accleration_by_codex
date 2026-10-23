"""Opt-in timing of the installed HF _sample; no per-stage synchronization."""
import ast
import contextlib
import inspect
import textwrap
import time
import torch
from transformers.generation.utils import GenerationMixin


class StageTimer:
    def __init__(self):
        self.rows = []
        self.step = -1

    @contextlib.contextmanager
    def span(self, stage):
        start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
        host = time.perf_counter()
        start.record()
        try:
            yield
        finally:
            end.record()
            self.rows.append(dict(step=self.step, stage=stage, host_start=host,
                                  host_ms=(time.perf_counter()-host)*1000,
                                  start=start, end=end))

    def condition(self, fn, *args, **kwargs):
        self.step += 1
        with self.span('scheduler'):
            result = fn(*args, **kwargs)
        return result

    def finish(self):
        # Caller has synchronized once after generate, never inside a stage.
        origin = self.rows[0]['start']
        return [dict(step=r['step'], stage=r['stage'], host_start=r['host_start'],
                     host_ms=r['host_ms'], device_start_ms=origin.elapsed_time(r['start']),
                     device_end_ms=origin.elapsed_time(r['end']),
                     device_ms=r['start'].elapsed_time(r['end'])) for r in self.rows]


def install(model, timer):
    source = textwrap.dedent(inspect.getsource(GenerationMixin._sample))
    tree = ast.parse(source)
    fn = tree.body[0]
    loop = next(n for n in fn.body if isinstance(n, ast.While))
    assert ast.unparse(loop.test).startswith('self._has_unfinished_sequences(')
    loop.test = ast.Call(func=ast.Attribute(ast.Name('_stage_timer', ast.Load()), 'condition', ast.Load()),
                         args=[loop.test.func, *loop.test.args], keywords=loop.test.keywords)

    def wrap(nodes, stage):
        return ast.With(items=[ast.withitem(ast.Call(
            ast.Attribute(ast.Name('_stage_timer', ast.Load()), 'span', ast.Load()),
            [ast.Constant(stage)], []), None)], body=nodes)

    first = loop.body[0]
    assert isinstance(first, ast.If) and ast.unparse(first.test) == 'prefill_consumed'
    assert len(first.body) == 3 and isinstance(first.body[-1], ast.With)
    first.body = [wrap(first.body[:2], 'scheduler'), first.body[-1]]
    ctx = first.body[-1]
    assert len(ctx.body) == 1 and 'model_forward(' in ast.unparse(ctx.body[0])
    ctx.body = [wrap(ctx.body, 'model_forward')]
    body = loop.body
    sample_start = next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and
                        any(isinstance(t,ast.Name) and t.id=='next_token_logits' for t in n.targets))
    select = next(i for i,n in enumerate(body) if isinstance(n,ast.If) and ast.unparse(n.test)=='do_sample')
    loop.body = [body[0], wrap(body[1:sample_start], 'postprocess'),
                 wrap(body[sample_start:select+1], 'sampling'),
                 wrap(body[select+1:], 'postprocess')]
    # Prefill precedes the loop in this HF version; preserve it verbatim.
    idx = next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and
               isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='self._prefill')
    fn.body[idx] = wrap([fn.body[idx]], 'prefill')
    ast.fix_missing_locations(tree)
    namespace = dict(GenerationMixin._sample.__globals__, _stage_timer=timer)
    exec(compile(tree, '<timed_hf_sample>', 'exec'), namespace)
    type(model)._sample = namespace['_sample']
    return source
