#!/bin/bash
HOSTFILE="${1:-~/.mlx/hostfile.json}"
echo "Verifying cluster..."
mlx.launch --hostfile "$HOSTFILE" -- /opt/homebrew/bin/python3 -c "
import mlx.core as mx
world = mx.distributed.init()
info = mx.device_info()
mem_gb = info.get('memory_size', 0) / (1024**3)
print(f'[Rank {world.rank()}/{world.size()}] {mem_gb:.0f}GB | Total: {mem_gb * world.size():.0f}GB')
"
