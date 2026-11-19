from __future__ import annotations

import hashlib
import os
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from typing import Any

__all__ = [
    "paper_provenance",
    "parse_core_breakdown",
    "parse_ioreg_gpu_core_count",
    "parse_system_profiler",
    "source_artifact",
    "software_provenance",
]


def source_artifact(
    path: str | Path,
    *,
    hash_max_bytes: int = 100_000_000,
) -> dict[str, Any]:
    """Provenance fingerprint for a benchmark input file.

    Records ``path``/``bytes``/``sha256`` so a committed benchmark result identifies the
    exact corpus it ran against. Consumed via ``benchmark.extra_info`` in the paper benches.
    """
    artifact_path = Path(path)
    if not artifact_path.exists():
        return {"path": str(artifact_path), "exists": False}
    if artifact_path.is_dir():
        return {
            "path": str(artifact_path),
            "exists": True,
            "type": "directory",
        }
    size = artifact_path.stat().st_size
    artifact: dict[str, Any] = {
        "path": str(artifact_path),
        "exists": True,
        "bytes": size,
    }
    if size <= hash_max_bytes:
        artifact["sha256"] = sha256_file(artifact_path)
    else:
        artifact["sha256"] = ""
        artifact["sha256_skipped"] = "file-too-large"
    return artifact


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paper_provenance(*, probe_metal_backend: bool = False) -> dict[str, Any]:
    return {
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "host_label": os.environ.get("BENCH_HOST_LABEL", ""),
        "command": [Path(sys.argv[0]).name, *sys.argv[1:]],
        "git": git_provenance(),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "node": platform.node(),
        },
        "cpu": cpu_provenance(),
        "environment": benchmark_environment(),
        "rdkit": rdkit_provenance(),
        "cuda": cuda_provenance(),
        "metal": metal_provenance(probe_backend=probe_metal_backend),
    }


def git_provenance() -> dict[str, Any]:
    commit = run_git("rev-parse", "HEAD")
    branch = run_git("rev-parse", "--abbrev-ref", "HEAD")
    status = run_git("status", "--short")
    short_commit = commit[:12] if commit else ""
    return {
        "commit": commit,
        "short_commit": short_commit,
        "branch": branch,
        "source_trees": {
            path: run_git("rev-parse", f"HEAD:{path}") for path in ("src", "tests")
        },
        "dirty": bool(status),
    }


def run_git(*args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    return completed.stdout.strip()


def repo_is_clean(paths: tuple[str, ...] = ("src", "tests")) -> bool:
    """True when the given paths have no uncommitted changes.

    Scoped to source paths so committing benchmark results under docs/ does not
    flag later runs as dirty (supports the run-run-run-commit workflow).
    """
    return run_git("status", "--porcelain", "--", *paths) == ""


def host_slug() -> str:
    """Combined cpu+gpu box identity, e.g. 'epyc-7763+rtx-4090', 'm5-pro'.

    Identifies the machine a benchmark ran on. The same box reports the same slug
    whether it runs cuda or rdkit, so cpu+gpu provenance is always available.
    """
    cpu = _slug(cpu_provenance().get("model"))
    gpu_name = cuda_provenance().get("device_name") or metal_provenance(
        probe_backend=False
    ).get("device_name")
    gpu = _slug(gpu_name)
    if not gpu or gpu == cpu:  # no discrete GPU, or Apple SoC where cpu == gpu
        return cpu or "unknown"
    return f"{cpu}+{gpu}"


_SLUG_NOISE = re.compile(
    r"\b(nvidia|geforce|amd|intel|core|cpu|processor|graphics|apple|radeon|with)\b",
    re.IGNORECASE,
)


def _slug(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"\(r\)|\(tm\)", " ", text)  # drop (R)/(TM)
    text = re.sub(r"@.*$", " ", text)  # drop "@ 2.50GHz" clock tail
    text = re.sub(r"\d+-core", " ", text)  # drop "64-Core"
    text = _SLUG_NOISE.sub(" ", text)
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text


def benchmark_environment() -> dict[str, str]:
    prefixes = (
        "BENCH_",
        "CUDA_",
        "PAPER_",
        "VESMA_",
        "VERTSY_",
    )
    names = (
        "XDG_CACHE_HOME",
        "UV_CACHE_DIR",
    )
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if key in names or any(key.startswith(prefix) for prefix in prefixes)
    }


def cpu_provenance() -> dict[str, Any]:
    logical_cores = env_int("BENCH_CPU_LOGICAL_CORES") or os.cpu_count()
    model = cpu_model()
    physical_cores = physical_core_count()
    breakdown = cpu_core_breakdown()
    usable_cores = usable_core_count(logical_cores)
    return {
        "model": model,
        "physical_cores": physical_cores,
        "logical_cores": logical_cores,
        "usable_cores": usable_cores,
        "core_breakdown": breakdown,
        "label": cpu_label(model, usable_cores),
        "total_ram_bytes": total_ram_bytes(),
    }


def cpu_model() -> str:
    configured = os.environ.get("BENCH_CPU_MODEL")
    if configured:
        return configured
    if sys.platform == "darwin":
        hardware = mac_hardware_info()
        chip = hardware.get("chip", "")
        if chip:
            return chip
        model = run_command("sysctl", "-n", "machdep.cpu.brand_string")
        if model:
            return model
    if sys.platform.startswith("linux"):
        model = linux_cpuinfo_value("model name")
        if model:
            return model
        hardware = linux_cpuinfo_value("Hardware")
        if hardware:
            return hardware
    return " ".join(part for part in (platform.processor(), platform.machine()) if part)


def physical_core_count() -> int | None:
    configured = env_int("BENCH_CPU_PHYSICAL_CORES")
    if configured is not None:
        return configured
    if sys.platform == "darwin":
        hardware = mac_hardware_info()
        core_count = hardware.get("cpu_core_count")
        if isinstance(core_count, int):
            return core_count
        value = run_command("sysctl", "-n", "hw.physicalcpu")
        return int(value) if value else None
    if sys.platform.startswith("linux"):
        parsed = linux_physical_cores()
        if parsed is not None:
            return parsed
    return None


def cpu_core_breakdown() -> str:
    if sys.platform != "darwin":
        return ""
    value = mac_hardware_info().get("cpu_core_breakdown", "")
    return value if isinstance(value, str) else ""


def cpu_label(model: str, usable_cores: int | None) -> str:
    if usable_cores is None:
        return model
    suffix = "CPU core" if usable_cores == 1 else "CPU cores"
    return f"{model}, {usable_cores} {suffix}"


def linux_physical_cores() -> int | None:
    path = Path("/proc/cpuinfo")
    if not path.exists():
        return None
    processors = 0
    pairs: set[tuple[str, str]] = set()
    physical_id = ""
    core_id = ""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            if physical_id and core_id:
                pairs.add((physical_id, core_id))
            physical_id = ""
            core_id = ""
            continue
        key, separator, value = line.partition(":")
        if not separator:
            continue
        key = key.strip()
        value = value.strip()
        if key == "processor":
            processors += 1
        elif key == "physical id":
            physical_id = value
        elif key == "core id":
            core_id = value
    if physical_id and core_id:
        pairs.add((physical_id, core_id))
    if pairs:
        return len(pairs)
    return processors or None


def usable_core_count(logical_cores: int | None) -> int | None:
    if hasattr(os, "sched_getaffinity"):
        try:
            return len(os.sched_getaffinity(0))
        except OSError:
            pass
    return logical_cores


def total_ram_bytes() -> int | None:
    configured = env_int("BENCH_RAM_BYTES")
    if configured is not None:
        return configured
    if sys.platform == "darwin":
        value = run_command("sysctl", "-n", "hw.memsize")
        if value:
            return int(value)
    if hasattr(os, "sysconf"):
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            pages = os.sysconf("SC_PHYS_PAGES")
        except OSError, ValueError:
            return None
        if isinstance(page_size, int) and isinstance(pages, int):
            return page_size * pages
    return None


@lru_cache
def mac_hardware_info() -> dict[str, Any]:
    if sys.platform != "darwin":
        return {}
    info = parse_system_profiler(
        run_command("system_profiler", "SPHardwareDataType", "SPDisplaysDataType")
    )
    if "gpu_core_count" not in info:
        ioreg_count = parse_ioreg_gpu_core_count(
            run_command("ioreg", "-r", "-c", "AGXAccelerator", "-d", "2")
        )
        if ioreg_count is not None:
            info["gpu_core_count"] = ioreg_count
    return info


def parse_system_profiler(text: str) -> dict[str, Any]:
    info: dict[str, Any] = {}
    section = ""
    in_apple_gpu = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.endswith(":") and not line.startswith(("Total ", "Model ")):
            section = line[:-1]
            in_apple_gpu = section.startswith("Apple ")
            continue
        key, separator, value = line.partition(":")
        if not separator:
            continue
        key = key.strip()
        value = value.strip()
        if section == "Hardware Overview":
            if key == "Chip":
                info["chip"] = value
            elif key == "Total Number of Cores":
                core_count, breakdown = parse_core_breakdown(value)
                info["cpu_core_count"] = core_count
                info["cpu_core_breakdown"] = breakdown
            elif key == "Memory":
                info["memory"] = value
        elif in_apple_gpu:
            if key == "Chipset Model":
                info["gpu_model"] = value
            elif key == "Total Number of Cores":
                core_count, _ = parse_core_breakdown(value)
                info["gpu_core_count"] = core_count
    return info


def parse_core_breakdown(value: str) -> tuple[int | None, str]:
    count_text, _, rest = value.partition("(")
    count = int(count_text.strip()) if count_text.strip().isdigit() else None
    breakdown = rest.rpartition(")")[0].strip() if rest else ""
    return count, breakdown


def parse_ioreg_gpu_core_count(text: str) -> int | None:
    match = re.search(r'"gpu-core-count"\s*=\s*(\d+)', text)
    return int(match.group(1)) if match else None


def env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return int(value)


def linux_cpuinfo_value(name: str) -> str:
    path = Path("/proc/cpuinfo")
    if not path.exists():
        return ""
    prefix = f"{name}\t:"
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(prefix):
            return line.partition(":")[2].strip()
    return ""


def run_command(*args: str) -> str:
    try:
        completed = subprocess.run(
            [*args],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    return completed.stdout.strip()


def rdkit_provenance() -> dict[str, Any]:
    try:
        import rdkit
        from rdkit import rdBase
    except ImportError:
        return {"available": False}
    return {
        "available": True,
        "version": str(rdkit.__version__),
        "rdkit_version": str(rdBase.rdkitVersion),
    }


def software_provenance() -> dict[str, Any]:
    """Exact Python package versions used by a publication benchmark run."""
    return {
        "vesma": version("vesma"),
        "numpy": version("numpy"),
        "pytest": version("pytest"),
        "pytest_benchmark": version("pytest-benchmark"),
        "rdkit": rdkit_provenance(),
    }


def metal_provenance(*, probe_backend: bool) -> dict[str, Any]:
    if sys.platform != "darwin":
        return {"available": False}
    hardware = mac_hardware_info()
    gpu_model = str(hardware.get("gpu_model") or hardware.get("chip") or "")
    gpu_cores = hardware.get("gpu_core_count")
    hardware_fields = {
        "device_name": gpu_model,
        "gpu_core_count": gpu_cores,
        "label": gpu_label(gpu_model, gpu_cores),
    }
    if not probe_backend and os.environ.get("BENCH_PROBE_METAL") != "1":
        return {"available": None, "probe_skipped": True} | hardware_fields
    try:
        from vesma.metal_matcher import probe_metal

        info = probe_metal()
    except Exception as exc:
        return {"available": False, "error": str(exc)} | hardware_fields
    device_name = info.name or gpu_model
    return {
        "available": True,
        "backend_type": info.backend_type,
        "device_name": device_name,
        "vendor": info.vendor,
        "description": info.description,
        "gpu_core_count": gpu_cores,
        "label": gpu_label(device_name, gpu_cores),
    }


def gpu_label(model: str, cores: Any) -> str:
    if isinstance(cores, int):
        suffix = "GPU core" if cores == 1 else "GPU cores"
        return f"{model}, {cores} {suffix}"
    return model


def cuda_provenance() -> dict[str, Any]:
    try:
        import cupy as cp
    except ImportError:
        return {"available": False}

    info: dict[str, Any] = {
        "available": bool(cp.cuda.is_available()),
        "cupy_version": str(cp.__version__),
    }
    if not info["available"]:
        return info

    try:
        device = cp.cuda.Device()
        attributes = device.attributes
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        properties = cp.cuda.runtime.getDeviceProperties(device.id)
        info.update(
            {
                "device_name": properties["name"].decode("utf-8"),
                "device_id": int(device.id),
                "total_global_mem_bytes": int(total_bytes),
                "free_mem_bytes": int(free_bytes),
                "total_global_mem": int(total_bytes),
                "free_mem": int(free_bytes),
                "multi_processor_count": int(attributes.get("MultiProcessorCount", 0)),
                "runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
                "driver_version": int(cp.cuda.runtime.driverGetVersion()),
            }
        )
    except Exception as exc:
        info["error"] = str(exc)
    return info
