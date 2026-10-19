"""Cache-controlled matmul, thread and stream sweeps with paired device timing."""
import json
import time
import torch
import torch_npu

torch.npu.set_device(0)
torch.manual_seed(42)
torch.npu.manual_seed_all(42)

def emit(row):
    with open('results/hardware_control.jsonl', 'a') as f:
        f.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)

with torch.inference_mode():
    weights = [torch.randn(5120, 17408, dtype=torch.bfloat16, device='npu') for _ in range(4)]
    for threads in [1, 8, 32]:
        torch.set_num_threads(threads)
        for batch in [1, 8, 32, 128]:
            for rotating in [False, True]:
                for stream_count in [1, 2]:
                    torch.npu.synchronize()
                    before = torch.npu.memory_allocated()
                    torch.npu.reset_peak_memory_stats()
                    xs = [torch.randn(batch, 5120, dtype=torch.bfloat16, device='npu') for _ in range(stream_count)]
                    ys = [torch.empty(batch, 17408, dtype=torch.bfloat16, device='npu') for _ in xs]
                    streams = [torch.npu.Stream() for _ in xs]
                    torch.npu.synchronize()
                    def run(n):
                        for i in range(n):
                            j = i % stream_count
                            with torch.npu.stream(streams[j]):
                                torch.mm(xs[j], weights[i % 4 if rotating else 0], out=ys[j])
                    run(8)
                    torch.npu.synchronize()
                    samples = []
                    for _ in range(5):
                        t = time.perf_counter()
                        run(40)
                        torch.npu.synchronize()
                        samples.append((time.perf_counter()-t)/40)
                    # Events bracket one stream independently; avoid interpreting cross-stream events.
                    start = torch.npu.Event(enable_timing=True)
                    end = torch.npu.Event(enable_timing=True)
                    with torch.npu.stream(streams[0]):
                        start.record()
                        for i in range(40):
                            torch.mm(xs[0], weights[i % 4 if rotating else 0], out=ys[0])
                        end.record()
                    end.synchronize()
                    emit(dict(threads=threads, batch=batch, rotating=rotating,
                              stream_count=stream_count, seconds=sorted(samples)[2], samples=samples,
                              single_stream_event_seconds=start.elapsed_time(end)/1000/40,
                              peak_increment_bytes=torch.npu.max_memory_allocated()-before))
                    del xs, ys, streams
