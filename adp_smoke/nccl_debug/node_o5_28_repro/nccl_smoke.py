"""Minimal NCCL health check: init + barrier + all_reduce across all local GPUs.
Healthy node: prints OK lines and exits 0 in <1 min. Faulty NCCL/P2P: hangs at
init or all_reduce until the 120s timeout aborts with an exception."""
import os, datetime, torch, torch.distributed as dist

rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=120))
x = torch.ones(1 << 20, device="cuda") * (rank + 1)
dist.barrier()
dist.all_reduce(x)
expected = sum(range(1, dist.get_world_size() + 1))
assert x[0].item() == expected, f"all_reduce wrong: {x[0].item()} != {expected}"
print(f"rank {rank}: NCCL OK on {torch.cuda.get_device_name(rank)}", flush=True)
dist.destroy_process_group()
