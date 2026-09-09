"""One launcher per allocated node; only rank zero runs the tuning driver."""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class Topology:
    nodes: int
    rank: int
    gpus: int


def topology(parameters: Mapping, env: Mapping[str, str]) -> Topology:
    # WORLD_SIZE is a process count under torchrun. Only interpret it as a
    # node count when the platform explicitly identifies a Pod rank.
    nodes = env.get("NNODES") or env.get("SLURM_NNODES")
    if nodes is None and "POD_RANK" in env:
        nodes = env.get("WORLD_SIZE")
    count = int(nodes if nodes is not None else parameters.get("trainer.nnodes", 1))
    rank = next((env[k] for k in ("NODE_RANK", "POD_RANK", "SLURM_NODEID") if k in env), None)
    if rank is None and nodes is not None:
        rank = env.get("RANK")
    if count > 1 and rank is None and env.get("RAY_CLUSTER_MODE") != "existing":
        raise ValueError("Multi-node launch requires NODE_RANK/POD_RANK (one launcher per node).")
    result = Topology(count, int(rank or 0), int(env.get("GPUS_PER_NODE", parameters.get("trainer.n_gpus_per_node", 8))))
    if result.nodes < 1 or not 0 <= result.rank < result.nodes or result.gpus < 1:
        raise ValueError(f"Invalid node topology: {result}")
    if int(env.get("LOCAL_WORLD_SIZE", "1")) > 1:
        raise ValueError("Run this launcher once per node, without torchrun.")
    return result


def ray_environment(env: Mapping[str, str]) -> dict[str, str]:
    result = dict(env)
    # Platform ranks describe Pods, not the GPU ranks assigned by verl.
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        result.pop(key, None)
    result.pop("RAY_ADDRESS", None)
    return result


def wait_for_head(host: str, port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            with socket.create_connection((host, port), timeout=min(3, timeout)):
                return
        except OSError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Ray head {host}:{port} unavailable after {timeout}s; start rank 0 and check networking.")
            time.sleep(2)


# Run readiness in a bounded subprocess: ray.init itself may block if the
# endpoint is reachable but GCS is not healthy.
READINESS = """
import ray, sys, time
ray.init(address=sys.argv[1])
expected, gpus = int(sys.argv[2]), int(sys.argv[3])
while True:
    nodes = [n for n in ray.nodes() if n['Alive'] and n.get('Resources', {}).get('GPU', 0) > 0]
    counts = [n['Resources']['GPU'] for n in nodes]
    print(f'[cluster] GPU nodes={len(nodes)}/{expected}, GPUs per node={counts}', flush=True)
    if len(nodes) > expected or any(c != gpus for c in counts):
        raise RuntimeError('Ray topology differs from requested allocation; check for stale/shared Ray nodes')
    if len(nodes) == expected:
        ray.shutdown()
        break
    time.sleep(5)
"""


def prepare_cluster(parameters: dict, *, dry_run: bool = False) -> bool:
    """Update runtime parameters. Return True only when this node runs the driver."""
    mode = os.getenv("RAY_CLUSTER_MODE", "managed")
    if mode not in {"managed", "existing"}:
        raise ValueError("RAY_CLUSTER_MODE must be managed or existing.")
    spec = topology(parameters, os.environ)
    parameters.update({"trainer.nnodes": spec.nodes, "trainer.n_gpus_per_node": spec.gpus})
    print(f"[cluster] nodes={spec.nodes}, rank={spec.rank}, GPUs/node={spec.gpus}, total={spec.nodes * spec.gpus}", flush=True)
    print(f"[cluster] mode={mode}", flush=True)
    if dry_run:
        print("[cluster] dry-run: no Ray processes started", flush=True)
        return spec.rank == 0
    if mode == "existing":
        if spec.rank != 0:
            print("[cluster] existing cluster: only rank 0 runs the driver", flush=True)
            return False
        host = os.getenv("RAY_HEAD_ADDR") or os.getenv("MASTER_ADDR")
        address = os.getenv("RAY_ADDRESS") or (f"{host}:{os.getenv('RAY_PORT', '6379')}" if host else "auto")
        if address == "local" or address.startswith("ray://"):
            raise ValueError("Existing mode requires a GCS host:port address or auto, not local/ray://.")
        print(f"[cluster] reusing {address}; checking GPU nodes", flush=True)
        verify_cluster(address, spec, ray_environment(os.environ))
        export_driver_address(address)
        return True
    if spec.nodes == 1:
        return True

    host = os.getenv("RAY_HEAD_ADDR") or os.getenv("MASTER_ADDR")
    if not host:
        raise ValueError("Multi-node launch requires MASTER_ADDR or RAY_HEAD_ADDR.")
    port = int(os.getenv("RAY_PORT", "6379"))
    timeout = float(os.getenv("RAY_START_TIMEOUT", "300"))
    if not 0 < port < 65536 or timeout <= 0:
        raise ValueError("RAY_PORT must be 1..65535 and RAY_START_TIMEOUT must be positive.")
    clean_env = ray_environment(os.environ)
    # C550's PyTorch build exposes the CUDA-compatible API. Do not advertise
    # nonexistent GPUs merely because --num-gpus was supplied to Ray.
    check = subprocess.run(
        [sys.executable, "-c", "import torch,sys; n=torch.cuda.device_count(); print('Visible GPUs:', n); sys.exit(0 if n == int(sys.argv[1]) else 1)", str(spec.gpus)],
        env=clean_env, timeout=60,
    )
    if check.returncode:
        raise RuntimeError("Visible GPU count differs from GPUS_PER_NODE/config; check the C550 environment and device allocation.")

    address = f"{host}:{port}"
    command = ["ray", "start", f"--num-gpus={spec.gpus}", "--disable-usage-stats"]
    if os.getenv("RAY_NODE_IP_ADDRESS"):
        command.append(f"--node-ip-address={os.environ['RAY_NODE_IP_ADDRESS']}")
    if spec.rank == 0:
        command.extend(["--head", f"--port={port}"])
        if not os.getenv("RAY_NODE_IP_ADDRESS"):
            command.append(f"--node-ip-address={socket.gethostbyname(host)}")
        subprocess.run(command, env=clean_env, check=True, timeout=timeout)
        verify_cluster(address, spec, clean_env)
        export_driver_address(address)
        return True

    wait_for_head(host, port, timeout)
    command.extend([f"--address={address}", "--block"])
    subprocess.run(command, env=clean_env, check=True)
    return False


def verify_cluster(address: str, spec: Topology, env: dict[str, str]) -> None:
    timeout = float(os.getenv("RAY_START_TIMEOUT", "300"))
    if timeout <= 0:
        raise ValueError("RAY_START_TIMEOUT must be positive.")
    try:
        subprocess.run(
            [sys.executable, "-u", "-c", READINESS, address, str(spec.nodes), str(spec.gpus)],
            env=env, check=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError("Timed out waiting for all Ray GPU nodes. Check the cluster address, node resources and worker logs.") from exc


def export_driver_address(address: str) -> None:
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        os.environ.pop(key, None)
    os.environ["RAY_ADDRESS"] = address
