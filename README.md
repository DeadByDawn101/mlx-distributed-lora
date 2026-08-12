<div align="center">

# Distributed LoRA Training on Apple Silicon

### The missing guide to multi-node LoRA fine-tuning with MLX + JACCL

<br>

**Train LLMs across multiple Macs over Thunderbolt 5.**
**Data parallelism. Gradient synchronization. Zero swap pressure.**

<br>

[![MLX](https://img.shields.io/badge/MLX-0.32.0+-blueviolet)](https://github.com/ml-explore/mlx)
[![License](https://img.shields.io/badge/license-Apache_2.0-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-Production-gold)]()

</div>

---

## Why This Exists

Apple shipped MLX distributed inference at WWDC 2026. Blog posts and videos showed large models running across multiple Macs. But nobody showed **training**.

This repository contains everything you need to run distributed LoRA fine-tuning across multiple Apple Silicon Macs connected via Thunderbolt 5 — the exact configuration steps, the data format requirements, the environment setup, and the launch commands that actually work.

**This is the first published guide to distributed LoRA training on an MLX/JACCL cluster.**

## What You Get

- **Distributed LoRA training** across 2+ Apple Silicon nodes
- **Data parallelism** with automatic gradient synchronization via `mlx.distributed`
- **Always-on networking** that survives reboots and cable swaps
- **Production hostfile configurations** for both Ring (TCP) and JACCL (RDMA) backends
- **Complete troubleshooting guide** covering every issue we encountered

## Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| macOS | 26.2 (Tahoe) | 26.4+ |
| MLX | 0.30.0 | 0.32.0 |
| Python | 3.10 | 3.13+ (via Homebrew) |
| Nodes | 2 | 2-4 |
| RAM per node | 32 GB | 128 GB |
| Interconnect | Thunderbolt 4 | Thunderbolt 5 |
| Cable | Direct TB cable | Not through hub/dock |

## Quick Start

### 1. Install MLX via Homebrew (both nodes)

System Python (3.9) ships with macOS but **MLX 0.30+ requires Python 3.10+**. Install via Homebrew:

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install MLX
brew install python mlx
/opt/homebrew/bin/pip3 install --break-system-packages mlx-lm datasets

# Verify
/opt/homebrew/bin/python3 -c "import mlx.core as mx; print(f'MLX {mx.__version__}')"
```

Repeat on **every node**. Versions must match.

### 2. Enable RDMA (both nodes)

Open **System Settings** → search **"RDMA"** → Enable **RDMA over Thunderbolt** → Reboot.

Verify after reboot:
```bash
ibv_devices
# Expected: rdma_en1, rdma_en2, rdma_en3
```

### 3. Connect Thunderbolt Cable

Plug a Thunderbolt 4 or 5 cable **directly** between the two Macs. Not through a hub, dock, or display.

For N nodes you need N×(N-1)/2 cables (full mesh):
- 2 nodes = 1 cable
- 3 nodes = 3 cables
- 4 nodes = 6 cables

### 4. Configure Networking

Run Apple's built-in configuration tool:

```bash
mlx.distributed_config \
  --verbose \
  --hosts user1@host1,user2@host2 \
  --over thunderbolt \
  --backend jaccl \
  --auto-setup \
  --output-hostfile ~/.mlx/hostfile.json
```

The tool will output exact `ifconfig` and `route` commands for each node. The key requirements it enforces:

| Setting | Value | Why |
|---------|-------|-----|
| Thunderbolt Bridge | **Down** | JACCL needs direct interface access |
| Subnet mask | `/30` (255.255.255.252) | Point-to-point link |
| Interface | `en3` (typical for MacBooks) | Direct Thunderbolt port |
| IPs | `192.168.0.1` / `192.168.0.2` | Apple's default assignment |

**Critical:** Bringing bridge0 down will kill any SSH session running through it. Configure the remote node via an alternative connection (WiFi, Tailscale, physical access) first.

<details>
<summary><strong>Manual setup (if auto-setup fails)</strong></summary>

**Node 0 (primary):**
```bash
sudo ifconfig en3 inet 192.168.0.1 netmask 255.255.255.252
sudo route change 192.168.0.2 -interface en3
```

**Node 1 (secondary):**
```bash
sudo ifconfig bridge0 down
sudo ifconfig en3 inet 192.168.0.2 netmask 255.255.255.252
sudo route change 192.168.0.1 -interface en3
```

Test: `ping -c 3 192.168.0.2`
</details>

### 5. Configure SSH (passwordless)

```bash
# Generate key
ssh-keygen -t ed25519 -f ~/.ssh/cluster -N ""

# Copy to all nodes (including self)
ssh-copy-id -i ~/.ssh/cluster user@localhost
ssh-copy-id -i ~/.ssh/cluster user@192.168.0.2

# Add to SSH config
cat >> ~/.ssh/config << 'EOF'
Host localhost
    IdentityFile ~/.ssh/cluster
    StrictHostKeyChecking no

Host 192.168.0.2
    User <remote_user>
    IdentityFile ~/.ssh/cluster
    StrictHostKeyChecking no
EOF

# Test (must work without password prompt)
ssh localhost "echo OK"
ssh 192.168.0.2 "echo OK"
```

### 6. Create Hostfile

```bash
mkdir -p ~/.mlx
cat > ~/.mlx/hostfile.json << 'EOF'
{
    "backend": "ring",
    "envs": ["MLX_METAL_FAST_SYNCH=1"],
    "hosts": [
        {"ssh": "user1@localhost", "ips": ["192.168.0.1"], "rdma": []},
        {"ssh": "user2@192.168.0.2", "ips": ["192.168.0.2"], "rdma": []}
    ]
}
EOF
```

### 7. Verify Cluster

```bash
mlx.launch --hostfile ~/.mlx/hostfile.json -- \
  /opt/homebrew/bin/python3 -c "
import mlx.core as mx
world = mx.distributed.init()
info = mx.device_info()
mem_gb = info.get('memory_size', 0) / (1024**3)
print(f'[Rank {world.rank()}/{world.size()}] {mem_gb:.0f}GB | Total: {mem_gb * world.size():.0f}GB')
"
```

**Expected output:**
```
[Rank 0/2] 128GB | Total: 256GB
[Rank 1/2] 128GB | Total: 256GB
```

### 8. Prepare Training Data

MLX-LM's local dataset loader expects this exact structure:

```
training-data/
├── train.jsonl      # Required
├── valid.jsonl      # Required
└── test.jsonl       # Optional
```

Each line is a JSON object with a `messages` array:

```json
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]}
```

**Critical format requirements:**
- Files must be `.jsonl` (not `.json`) — MLX-LM's `load_local_dataset()` hardcodes this extension
- Validation file must be named `valid.jsonl` (not `validation.jsonl`) — MLX-LM looks for the split name `"valid"`, but HuggingFace's `datasets` library auto-maps filenames to `"validation"`. Using `.jsonl` with the local loader bypasses this entirely.
- Training data must exist at the **same absolute path** on all nodes

Sync data to all nodes:
```bash
rsync -avz /path/to/training-data/ user@192.168.0.2:/path/to/training-data/
```

### 9. Sync Model to All Nodes

Each node needs a local copy of the model. Sync via Thunderbolt (fastest):

```bash
rsync -avz ~/.cache/huggingface/hub/models--<your-model>/ \
  user@192.168.0.2:~/.cache/huggingface/hub/models--<your-model>/
```

Or let each node download independently (slower but simpler):
```bash
ssh user@192.168.0.2 "/opt/homebrew/bin/python3 -c \"from huggingface_hub import snapshot_download; snapshot_download('mlx-community/Qwen3-8B-4bit')\""
```

### 10. Launch Distributed Training

```bash
mlx.launch --hostfile ~/.mlx/hostfile.json -- \
  /opt/homebrew/bin/python3 -m mlx_lm lora \
  --model mlx-community/Qwen3-8B-4bit \
  --data /path/to/training-data \
  --train \
  --adapter-path /path/to/output-adapters \
  --iters 1000 \
  --batch-size 2 \
  --num-layers 12 \
  --learning-rate 1e-5 \
  --steps-per-eval 200 \
  --steps-per-report 25 \
  --max-seq-length 2048 \
  --seed 42 \
  --save-every 200 \
  --grad-checkpoint
```

**What distributed training changes:**
- `--batch-size 2` → one sample per node, effectively doubled throughput
- `--grad-checkpoint` → reduces per-node memory for larger models
- Gradients are automatically synchronized via `mx.distributed.all_sum()` at each step
- Each node processes a different slice of the dataset (data parallelism)

**Expected output:**
```
Node 0 of 2
Node 1 of 2
Iter 1: Val loss 3.746, Val took 1.295s
Iter 25: Train loss 2.122, Learning Rate 1.000e-05, It/sec 1.090, Tokens/sec 623.840
```

---

## Always-On Networking

Static IPs reset on reboot and cable swap. Install a LaunchDaemon to auto-restore them:

```bash
# Copy the restore script
sudo cp scripts/ip-restore.sh /usr/local/bin/
sudo chmod +x /usr/local/bin/ip-restore.sh

# Set node role (0 = primary, 1 = secondary)
echo "NODE_ROLE=0" | sudo tee /etc/mlx-cluster.conf

# Install and load the LaunchDaemon
sudo cp services/com.mlx-cluster.ip-restore.plist /Library/LaunchDaemons/
sudo launchctl bootstrap system /Library/LaunchDaemons/com.mlx-cluster.ip-restore.plist
```

Repeat on each node with the appropriate `NODE_ROLE`.

---

## Backends

| Backend | Transport | Latency | Bandwidth | Status |
|---------|-----------|---------|-----------|--------|
| `ring` | TCP sockets | ~500 µs | ~10 Gbps | Production-ready |
| `jaccl` | RDMA verbs | 5-9 µs | 50-60 Gbps | Provider bugs (see below) |

### Ring Backend (Recommended)

Works immediately. Uses TCP over Thunderbolt. 80 Gbps pipe, ~10 Gbps effective for gradient sync. Sufficient for LoRA training on models up to 70B+ parameters.

### JACCL RDMA Backend (Experimental)

Lower latency, higher bandwidth. Requires RDMA enabled on all nodes. Currently has provider-level bugs in macOS that cause initialization failures:

- **Protection Domain resource leak** after ~60 transfers
- **Memory Region cap** at 100 per host
- **Queue Pair degradation** after ~23 minutes idle

These are documented in the Apple RDMA provider and affect all JACCL users. Ring backend is the reliable alternative until Apple patches the provider.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `No module named 'mlx'` on remote node | Install MLX via Homebrew: `brew install mlx` |
| `No module named 'datasets'` | `/opt/homebrew/bin/pip3 install --break-system-packages datasets` |
| `FileNotFoundError: Couldn't find any data file` | Rename `.json` → `.jsonl`. MLX-LM requires `.jsonl` extension. |
| `Training set not found or empty` | Ensure `train.jsonl` exists (not `train.json`). Check `valid.jsonl` (not `validation.jsonl`). |
| `Rank 0/1` instead of `Rank 0/2` | Nodes aren't linking. Check SSH, check IPs match hostfile. |
| SSH password prompt | Run `ssh-copy-id` to the target. Add to `~/.ssh/config`. |
| `bridge0 down` kills SSH | Configure remote node via WiFi or Tailscale before bringing bridge down. |
| Different Python versions | Both nodes must use the same Python. Use `/opt/homebrew/bin/python3` everywhere. |
| System Python 3.9 blocks MLX 0.30+ | MLX 0.30+ requires Python 3.10+. Install via Homebrew. |
| IPs reset after reboot | Install the always-on LaunchDaemon (see above). |
| `mlx.distributed_config` not found | It's a standalone binary at `/opt/homebrew/bin/mlx.distributed_config`, not a Python module. |

---

## Architecture

```
┌──────────────────────────────────────────────┐
│            mlx.launch (orchestrator)         │
│    Reads hostfile, SSHs to all nodes,        │
│    launches same command on each              │
├──────────────────────────────────────────────┤
│                                              │
│  ┌─────────────┐       ┌─────────────┐      │
│  │   Node 0    │  TB5  │   Node 1    │      │
│  │  128 GB     │◄─────►│  128 GB     │      │
│  │  Rank 0/2   │80Gbps │  Rank 1/2   │      │
│  └──────┬──────┘       └──────┬──────┘      │
│         │                     │              │
│         ▼                     ▼              │
│    Forward pass          Forward pass        │
│    Compute loss          Compute loss        │
│    Backprop              Backprop            │
│         │                     │              │
│         └────────┬────────────┘              │
│                  ▼                           │
│         mx.distributed.all_sum()             │
│         (gradient synchronization)           │
│                  │                           │
│                  ▼                           │
│         Optimizer step (both nodes)          │
│         Weights updated identically          │
└──────────────────────────────────────────────┘
```

---

## Performance Notes

- **Ring backend** adds ~5-10% overhead for gradient sync on typical LoRA training
- **`MLX_METAL_FAST_SYNCH=1`** is critical — without it, inference is 5-6x slower
- **`--grad-checkpoint`** trades compute for memory — essential for large models
- **`mx.set_wired_limit(N)`** prevents swap pressure during training
- Same-model training on 2 nodes ≈ 1.8-1.9x single-node throughput (near-linear scaling)

---

## Acknowledgments

Built on [MLX](https://github.com/ml-explore/mlx) by Apple Machine Learning Research. Networking informed by research into [JACCL provider behavior](https://github.com/tmc/gojaccl) and the broader Apple Silicon ML community.

## License

Apache 2.0

---

<div align="center">

**Built by [RavenX AI Labs](https://github.com/DeadByDawn101)**

*First published distributed LoRA training on Apple Silicon JACCL clusters.*
*August 9, 2026.*

</div>

---

## Production Learnings (from extended testing)

### Path Rules for Distributed Training

| Do | Don't | Why |
|----|-------|-----|
| `/opt/training` | `~/training` | `~` expands to different home dirs per node |
| `/var/tmp/adapters` | `/tmp/adapters` | `/tmp` is cleared on reboot |
| Absolute paths only | Relative paths | mlx.launch SSHs to remote — relative paths resolve differently |

### Validation Settings

Long validation causes TCP timeouts. The ring backend's TCP connection drops if nodes are silent for too long during a synchronization pause.

```bash
# SAFE: fast validation, infrequent evaluation
--val-batches 5 --steps-per-eval 500

# DANGEROUS: slow validation, frequent evaluation  
--val-batches 25 --steps-per-eval 200  # 52 seconds of silence = TCP death
```

### SSH Persistence

Add to `~/.ssh/config` on the primary node:

```
Host *
    ControlMaster auto
    ControlPath /tmp/ssh-%r@%h:%p
    ControlPersist 600
    ServerAliveInterval 30
    ServerAliveCountMax 10
    TCPKeepAlive yes
```

### Use `networksetup` Not `ifconfig`

Raw `ifconfig` sets IPs at the kernel level. macOS configd doesn't know about them and will fight you with DHCP. Use `networksetup -setmanual` instead — this creates a macOS-managed network service that survives sleep/wake and integrates with the system.

```bash
# WRONG (configd will fight it):
sudo ifconfig en3 192.168.0.1 netmask 255.255.255.252

# RIGHT (macOS manages it):
sudo networksetup -setmanual "JACCL" 192.168.0.1 255.255.255.252
```

### Do NOT use WatchPaths LaunchDaemon with `ifconfig`

A LaunchDaemon watching `/Library/Preferences/SystemConfiguration` that calls `ifconfig` creates an infinite loop: ifconfig triggers configd → writes to SystemConfiguration → fires WatchPaths → runs script → calls ifconfig → loop. This will destabilize training.

---

## Thunderbolt Kernel Panic Fix (macOS 26.6.1)

If plugging in a TB5 cable causes kernel panics (`far: 0x0`, pid `route` or `configd`), run this on **both** nodes:

```bash
sudo defaults write /Library/Preferences/SystemConfiguration/com.apple.autonetworkservices AutomaticServiceCreation -bool false
```

This stops macOS from auto-creating network services on Thunderbolt interfaces. The race condition between `configd/route` and PCIe driver initialization causes a NULL pointer dereference. See [our detailed write-up](https://github.com/DeadByDawn101/RavenX-JACCL-MLX/blob/main/docs/KERNEL-PANIC-FIX.md) for the full analysis (7 kernel panics, root-caused and fixed).

### Additional Critical Rules

- **No Thunderbolt Ethernet adapters** on either Mac (NCM/ECM driver bleeds across TB5 peer bus)
- **SSH over WiFi, data over TB5** — two separate network planes
- **TCP keepalive=1** on both nodes (`sysctl -w net.inet.tcp.always_keepalive=1`)
- **Free en3 from bridge0** after every boot on macOS 26.6.1 (`sudo ifconfig bridge0 deletem en3`)

### Dual-Plane Architecture

```
Management: WiFi (192.168.1.x) — SSH, coordination
Data:       TB5  (192.168.0.x) — gradient sync
```

Hostfile uses WiFi IPs for SSH, TB5 IPs for data:
```json
{
    "backend": "ring",
    "envs": ["MLX_METAL_FAST_SYNCH=1"],
    "hosts": [
        {"ssh": "user0@192.168.1.155", "ips": ["192.168.0.1"], "rdma": []},
        {"ssh": "user1@192.168.1.165", "ips": ["192.168.0.2"], "rdma": []}
    ]
}
```
