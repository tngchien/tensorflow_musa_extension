"""Common helpers for simple MUSA operator benchmarks."""

import argparse
import csv
import json
import re
import shlex
import shutil
import subprocess
import sys
from functools import wraps
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2


class HardwareSpec:
    PASS_RATIO = 0.70
    SPECS = {
        "PH1": {
            "memory_bandwidth_gbps": 1600.0,
            "peak_tflops": {
                "float32": 31.3,
                "float16": 31.3,
                "bfloat16": 31.3,
                "float64": 15.6,
                "int32": 15.6,
                "int8": 125.2,
            },
            "tensor_peak_tflops": {
                "float16": 500.7,
                "bfloat16": 500.7,
                "tf32": 250.3,
                "int8": 1001.4,
            },
            "local_memory_per_mp_kb": 192,
            "l2_cache_per_mpc_mb": 1.5,
            "l3_cache_per_gpu_mb": 60,
        }
    }

    def __init__(self, arch="PH1"):
        self.arch = arch
        self.spec = self.SPECS[arch]

    def memory_bandwidth_gbps(self):
        return self.spec["memory_bandwidth_gbps"]

    def peak_tflops(self, dtype_name):
        return self.spec["peak_tflops"].get(dtype_name, 0.0)

    def evaluate(self, result):
        p50_s = result["p50_ms"] / 1000.0
        effective_gbps = (
            result["effective_bytes"] / p50_s / 1e9 if p50_s else 0.0
        )
        effective_tflops = (
            result["effective_ops"] / p50_s / 1e12 if p50_s else 0.0
        )
        peak_tflops = self.peak_tflops(result["dtype"])
        peak_bandwidth = self.memory_bandwidth_gbps()
        compute_util = (
            effective_tflops / peak_tflops * 100.0 if peak_tflops else 0.0
        )
        bandwidth_util = effective_gbps / peak_bandwidth * 100.0

        result["effective_gbps"] = effective_gbps
        result["effective_tflops"] = effective_tflops
        result["compute_utilization_pct"] = compute_util
        result["bandwidth_utilization_pct"] = bandwidth_util
        result["peak_tflops"] = peak_tflops
        result["peak_memory_bandwidth_gbps"] = peak_bandwidth
        result["meets_70pct_compute"] = compute_util >= self.PASS_RATIO * 100.0
        result["meets_70pct_bandwidth"] = bandwidth_util >= self.PASS_RATIO * 100.0
        result["status"] = "PASS" if result["meets_70pct_bandwidth"] else "MISS"
        return result


class MCUProfiler:
    DEFAULT_SECTIONS = "SpeedOfLight,LaunchStats,MemoryWorkloadAnalysis"

    def __init__(self, args, op_name, script_path):
        self.args = args
        self.op_name = op_name
        self.script_path = Path(script_path)
        self.exe = shutil.which("mcu")

    def run(self, shape_index, dtype_name):
        if not self.exe:
            return {"status": "skipped", "reason": "mcu not found"}

        case_dir = Path(self.args.profile_dir) / self.op_name / str(shape_index) / dtype_name
        export_dir = case_dir / "export"
        report_path = case_dir / "report.mcu-rep"
        case_dir.mkdir(parents=True, exist_ok=True)
        export_dir.mkdir(parents=True, exist_ok=True)

        profile_command = self.command(shape_index, dtype_name, report_path)
        profile = subprocess.run(
            profile_command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        export_command = self.export_command(report_path, export_dir)
        export = subprocess.run(
            export_command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        metrics = self.parse_metrics(export_dir)
        self._parse_text_content(profile.stdout, metrics)
        self._parse_text_content(export.stdout, metrics)
        summary = self.summarize(metrics)
        profile_error = "==ERROR==" in (profile.stdout + profile.stderr)
        export_error = "==ERROR==" in (export.stdout + export.stderr)
        ok = (
            profile.returncode == 0 and export.returncode == 0 and
            not profile_error and not export_error and report_path.exists()
        )
        return {
            "status": "ok" if ok else "failed",
            "profile_command": profile_command,
            "export_command": export_command,
            "profile_returncode": profile.returncode,
            "export_returncode": export.returncode,
            "report": str(report_path),
            "export_dir": str(export_dir),
            "profile_stdout": profile.stdout,
            "profile_stderr": profile.stderr,
            "export_stdout": export.stdout,
            "export_stderr": export.stderr,
            **summary,
        }

    def command(self, shape_index, dtype_name, report_path):
        command = [
            self.exe,
            "-f",
            "-o",
            str(report_path),
            "--sections",
            self.args.mcu_sections,
        ]
        if self.args.mcu_extra_args:
            command.extend(shlex.split(self.args.mcu_extra_args))
        command.extend([
            sys.executable,
            str(self.script_path),
            "--single-index",
            str(shape_index),
            "--single-dtype",
            dtype_name,
            "--warmup",
            str(self.args.warmup),
            "--iters",
            str(self.args.iters),
            "--repeat",
            "1",
            "--profile",
            "none",
            "--profile-child",
        ])
        if self.args.validate:
            command.append("--validate")
        return command

    def export_command(self, report_path, export_dir):
        return [
            self.exe,
            "-f",
            "--export-pfm",
            "--export-report",
            str(report_path),
            "--export-output",
            str(export_dir),
        ]

    def parse_metrics(self, export_dir):
        raw = {}
        for path in Path(export_dir).rglob("*"):
            if not path.is_file() or path.suffix.lower() not in (".csv", ".json", ".txt", ".log"):
                continue
            try:
                if path.suffix.lower() == ".json":
                    self._parse_json(path, raw)
                elif path.suffix.lower() == ".csv":
                    self._parse_csv(path, raw)
                else:
                    self._parse_text(path, raw)
            except Exception as exc:
                raw[f"parse_error:{path}"] = str(exc)
        return raw

    def _parse_json(self, path, raw):
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            data = json.load(handle)
        self._flatten_json(data, "", raw)

    def _flatten_json(self, value, prefix, raw):
        if isinstance(value, dict):
            for key, child in value.items():
                name = f"{prefix}.{key}" if prefix else str(key)
                self._flatten_json(child, name, raw)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                self._flatten_json(child, f"{prefix}.{index}", raw)
        else:
            number = parse_number(value)
            if number is not None:
                raw[normalize_metric(prefix)] = number

    def _parse_csv(self, path, raw):
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            rows = list(csv.reader(handle))
        if not rows:
            return
        header = [cell.strip().lower() for cell in rows[0]]
        metric_idx = find_header_index(header, ("metric name", "metric", "name"))
        value_idx = find_header_index(header, ("metric value", "value", "avg", "average"))
        if metric_idx is not None and value_idx is not None:
            for row in rows[1:]:
                if len(row) <= max(metric_idx, value_idx):
                    continue
                key = row[metric_idx].strip()
                value = parse_number(row[value_idx])
                if key and value is not None:
                    raw[normalize_metric(key)] = value
            return
        for row in rows:
            if len(row) >= 2:
                key = row[0].strip()
                value = parse_number(row[1])
                if key and value is not None:
                    raw[normalize_metric(key)] = value

    def _parse_text(self, path, raw):
        self._parse_text_content(path.read_text(encoding="utf-8", errors="ignore"), raw)

    def _parse_text_content(self, text, raw):
        for line in text.splitlines():
            match = re.match(r"\s*([^:=,]{3,}?)\s*[:=,]\s*([-+0-9.eE%]+)", line)
            if match:
                raw[normalize_metric(match.group(1))] = parse_number(match.group(2))
                continue
            columns = re.match(
                r"\s*([A-Za-z][A-Za-z0-9_()/# .-]*?)\s+(?:%|us|cycle|Gbyte/s|Ghz|thread|register/thread|Kbyte/block|byte/block)?\s+([-+0-9.,]+)\s*$",
                line,
            )
            if columns:
                key = columns.group(1).strip()
                value = parse_number(columns.group(2))
                if key and value is not None:
                    raw[normalize_metric(key)] = value

    def summarize(self, metrics):
        bank_value = find_metric(metrics, (r"bank.*conflict", r"conflict.*bank", r"bank_conflict"))
        l1_hit = find_metric(metrics, (r"^l1__sector_hit_rate\.pct$", r"l1.*hit.*rate"))
        l2_hit = find_metric(metrics, (r"l2.*hit.*rate",))
        key_metrics = {
            "ddr_util_pct": find_metric(metrics, (r"ddr__throughput.*pct", r"ddr_throughput")),
            "gpu_memory_util_pct": find_metric(metrics, (r"gpu__compute_memory_throughput.*pct", r"^memory_throughput$")),
            "gpu_memory_access_util_pct": find_metric(metrics, (r"gpu__compute_memory_access_throughput.*pct",)),
            "gpu_memory_request_util_pct": find_metric(metrics, (r"gpu__compute_memory_request_throughput.*pct",)),
            "l1_throughput_pct": find_metric(metrics, (r"l1__throughput.*pct", r"l1_cache_throughput")),
            "l2_throughput_pct": find_metric(metrics, (r"l2pu__throughput.*pct", r"l2_cache_throughput")),
            "duration_us": find_metric(metrics, (r"^duration$",)),
        }
        key_metrics = {key: value for key, value in key_metrics.items() if value is not None}
        return {
            "bank_conflict": {
                "value": bank_value,
                "severity": bank_conflict_severity(bank_value),
            },
            "l1_hit_rate_pct": l1_hit,
            "l2_hit_rate_pct": l2_hit,
            "key_metrics": key_metrics,
            "raw_metric_count": len(metrics),
            "raw_metrics": metrics,
        }


def normalize_metric(name):
    return re.sub(r"\s+", "_", str(name).strip().lower())


def find_header_index(header, names):
    for name in names:
        if name in header:
            return header.index(name)
    return None


def parse_number(value):
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", str(value))
    return float(match.group(0)) if match else None


def find_metric(metrics, patterns):
    for pattern in patterns:
        regex = re.compile(pattern)
        for key, value in metrics.items():
            if regex.search(key):
                return value
    return None


def bank_conflict_severity(value):
    if value is None:
        return "unknown"
    if value < 1.0:
        return "none"
    if value < 5.0:
        return "low"
    if value < 15.0:
        return "medium"
    return "high"


def attach_perf_metrics(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        return HardwareSpec().evaluate(result)
    return wrapper


def profile_with_mcu(op_name, script_path):
    def decorator(fn):
        @wraps(fn)
        def wrapper(lhs_shape, rhs_shape, dtype_info, args, seed, shape_index):
            result = fn(lhs_shape, rhs_shape, dtype_info, args, seed, shape_index)
            if args.profile == "mcu" and not args.profile_child:
                result["mcu"] = MCUProfiler(args, op_name, script_path).run(
                    shape_index, result["dtype"])
            return result
        return wrapper
    return decorator


def create_config():
    config = config_pb2.ConfigProto()
    config.allow_soft_placement = True
    rewrite = config.graph_options.rewrite_options
    rewrite.min_graph_nodes = -1
    rewrite.constant_folding = rewriter_config_pb2.RewriterConfig.OFF
    rewrite.arithmetic_optimization = rewriter_config_pb2.RewriterConfig.OFF
    rewrite.dependency_optimization = rewriter_config_pb2.RewriterConfig.OFF
    rewrite.function_optimization = rewriter_config_pb2.RewriterConfig.OFF
    rewrite.shape_optimization = rewriter_config_pb2.RewriterConfig.OFF
    return config


def num_elements(shape):
    return int(np.prod(shape, dtype=np.int64)) if shape else 1


def broadcast_shape(lhs, rhs):
    lhs = list(lhs)
    rhs = list(rhs)
    rank = max(len(lhs), len(rhs))
    lhs = [1] * (rank - len(lhs)) + lhs
    rhs = [1] * (rank - len(rhs)) + rhs
    out = []
    for lhs_dim, rhs_dim in zip(lhs, rhs):
        if lhs_dim == rhs_dim or lhs_dim == 1 or rhs_dim == 1:
            out.append(max(lhs_dim, rhs_dim))
        else:
            raise ValueError(f"Incompatible shapes: {lhs} vs {rhs}")
    return out


def make_data(shape, dtype_name, np_dtype, rng):
    if dtype_name.startswith("int"):
        return rng.randint(-16, 16, size=shape).astype(np_dtype)
    return rng.uniform(-1.0, 1.0, size=shape).astype(np_dtype)


def validate_output(actual, expected, dtype_name):
    if dtype_name.startswith("int"):
        if not np.array_equal(actual, expected):
            raise AssertionError("int result mismatch")
        return
    rtol = 1e-2 if dtype_name in ("float16", "bfloat16") else 1e-5
    atol = 1e-2 if dtype_name in ("float16", "bfloat16") else 1e-8
    if not np.allclose(actual.astype(np.float32), expected.astype(np.float32),
                       rtol=rtol, atol=atol):
        raise AssertionError("float result mismatch")


def print_header():
    print(
        f"{'dtype':<8} {'lhs':<18} {'rhs':<18} {'out':<18} "
        f"{'p50_ms':>9} {'TFLOPS':>9} {'compute%':>9} "
        f"{'GB/s':>9} {'bw%':>8} {'status':>7}"
    )


def print_result(result):
    print(
        f"{result['dtype']:<8} {str(result['lhs_shape']):<18} "
        f"{str(result['rhs_shape']):<18} {str(result['output_shape']):<18} "
        f"{result['p50_ms']:>9.3f} {result['effective_tflops']:>9.3f} "
        f"{result['compute_utilization_pct']:>9.1f} "
        f"{result['effective_gbps']:>9.1f} "
        f"{result['bandwidth_utilization_pct']:>8.1f} {result['status']:>7}"
    )
    mcu = result.get("mcu")
    if mcu:
        key_metrics = mcu.get("key_metrics", {})
        print(
            "  MCU: "
            f"status={mcu.get('status')} "
            f"bank_conflict={mcu.get('bank_conflict', {}).get('severity')} "
            f"l1_hit={format_optional_pct(mcu.get('l1_hit_rate_pct'))} "
            f"l2_hit={format_optional_pct(mcu.get('l2_hit_rate_pct'))} "
            f"ddr_util={format_optional_pct(key_metrics.get('ddr_util_pct'))} "
            f"mem_util={format_optional_pct(key_metrics.get('gpu_memory_util_pct'))}"
        )


def format_optional_pct(value):
    return "unknown" if value is None else f"{value:.1f}%"


def add_common_args(parser):
    parser.add_argument("--device", default="/device:MUSA:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--json", default="")
    parser.add_argument("--profile", choices=("none", "mcu"), default="none")
    parser.add_argument("--profile-dir", default="benchmark/profiles")
    parser.add_argument("--mcu-sections", default=MCUProfiler.DEFAULT_SECTIONS)
    parser.add_argument("--mcu-extra-args", default="")
    parser.add_argument("--profile-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--single-index", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--single-dtype", default="", help=argparse.SUPPRESS)


def load_musa_or_fail(tf_musa):
    if not tf_musa.is_plugin_loaded():
        tf_musa.load_plugin()
    if not tf.config.list_physical_devices("MUSA"):
        raise RuntimeError("No MUSA devices found")


def save_json(path, results):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump({"hardware": HardwareSpec.SPECS, "results": results}, handle, indent=2)
    print(f"\nSaved JSON summary to: {output}")
