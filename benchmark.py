import time
import torch
from ebmlstm import EventAugmentedLSTM


def _percentile(sorted_vals, p):
    idx = int(len(sorted_vals) * p)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


def benchmark(model, x, n_warmup=100, n_iters=1000):
    device = x.device.type
    model.eval()

    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()

        latencies = []
        if device == "cuda":
            for _ in range(n_iters):
                t_start = torch.cuda.Event(enable_timing=True)
                t_end   = torch.cuda.Event(enable_timing=True)
                t_start.record()
                model(x)
                t_end.record()
                torch.cuda.synchronize()
                latencies.append(t_start.elapsed_time(t_end) * 1e3)  # ms → µs
        else:
            for _ in range(n_iters):
                t0 = time.perf_counter()
                model(x)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1e6)  # s → µs

    return latencies


def report(label, latencies):
    sv = sorted(latencies)
    mean = sum(sv) / len(sv)
    p50  = _percentile(sv, 0.50)
    p95  = _percentile(sv, 0.95)
    p99  = _percentile(sv, 0.99)
    print(f"\n=== {label} ===")
    print(f"  Mean : {mean:>10.1f} µs")
    print(f"  P50  : {p50:>10.1f} µs")
    print(f"  P95  : {p95:>10.1f} µs")
    print(f"  P99  : {p99:>10.1f} µs")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    print(f"Warmup : 100 passes  |  Timed : 1000 passes")

    T, B, input_dim = 500, 32, 16
    x = torch.randn(T, B, input_dim, device=device)

    model = EventAugmentedLSTM(
        input_dim=input_dim,
        mem_slots=8,
        hidden_dim=64,
        out_dim=1,
    ).to(device)

    latencies = benchmark(model, x, n_warmup=100, n_iters=1000)
    report("New Implementation (Parallel Weighted Aggregator)", latencies)
