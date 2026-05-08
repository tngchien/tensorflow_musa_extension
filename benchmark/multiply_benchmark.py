#!/usr/bin/env python3
"""Simple smoke benchmark for the MUSA Multiply operator."""

import argparse
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import tensorflow_musa as tf_musa

from commons import (
    HardwareSpec,
    add_common_args,
    attach_perf_metrics,
    broadcast_shape,
    create_config,
    load_musa_or_fail,
    make_data,
    num_elements,
    print_header,
    print_result,
    profile_with_mcu,
    save_json,
    validate_output,
)


tf1.disable_eager_execution()

DTYPES = [
    ("float32", tf.float32, np.float32, 4),
    ("float16", tf.float16, np.float16, 2),
    ("bfloat16", tf.bfloat16, np.float32, 2),
    ("int32", tf.int32, np.int32, 4),
]

SHAPES = [
    ([3], [3]),
    ([1024], []),
    ([1024 * 1024], [1024 * 1024]),
    ([1024, 1024], [1024, 1024]),
    ([1024, 1024], [1024]),
    ([1024, 1], [1, 1024]),
    ([5, 1, 3], [1, 7, 3]),
    ([1, 1, 10], [5, 3, 10]),
]


def build_graph(lhs_shape, rhs_shape, dtype_name, tf_dtype, np_dtype, device, seed):
    rng = np.random.RandomState(seed)
    lhs_np = make_data(lhs_shape, dtype_name, np_dtype, rng)
    rhs_np = make_data(rhs_shape, dtype_name, np_dtype, rng)
    expected = np.multiply(lhs_np, rhs_np)

    graph = tf.Graph()
    with graph.as_default():
        with tf.device(device):
            lhs = tf.constant(lhs_np, dtype=tf_dtype, name="lhs")
            rhs = tf.constant(rhs_np, dtype=tf_dtype, name="rhs")
            output = tf.math.multiply(lhs, rhs, name="multiply")
            output = tf.identity(output, name="output")
            with tf.control_dependencies([output]):
                step = tf.no_op(name="step")
    return graph, step, output, expected


@profile_with_mcu("multiply", Path(__file__).resolve())
@attach_perf_metrics
def run_one_case(lhs_shape, rhs_shape, dtype_info, args, seed, shape_index):
    dtype_name, tf_dtype, np_dtype, dtype_bytes = dtype_info
    graph, step, output, expected = build_graph(
        lhs_shape, rhs_shape, dtype_name, tf_dtype, np_dtype, args.device, seed
    )

    times = []
    with tf1.Session(graph=graph, config=create_config()) as sess:
        for _ in range(args.warmup):
            sess.run(step)
        if args.validate:
            validate_output(sess.run(output), expected, dtype_name)
        for _ in range(args.repeat):
            start = time.perf_counter()
            for _ in range(args.iters):
                sess.run(step)
            times.append((time.perf_counter() - start) / args.iters)

    out_shape = broadcast_shape(lhs_shape, rhs_shape)
    output_elements = num_elements(out_shape)
    p50_ms = float(np.percentile(times, 50) * 1000.0)
    p90_ms = float(np.percentile(times, 90) * 1000.0)
    min_ms = float(np.min(times) * 1000.0)

    return {
        "dtype": dtype_name,
        "lhs_shape": lhs_shape,
        "rhs_shape": rhs_shape,
        "output_shape": out_shape,
        "output_elements": output_elements,
        "p50_ms": p50_ms,
        "p90_ms": p90_ms,
        "min_ms": min_ms,
        "effective_ops": output_elements,
        "effective_bytes": output_elements * dtype_bytes * 3,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Simple Multiply smoke benchmark.")
    add_common_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    load_musa_or_fail(tf_musa)

    shape_items = list(enumerate(SHAPES))
    if args.single_index >= 0:
        shape_items = [(args.single_index, SHAPES[args.single_index])]

    dtypes = DTYPES
    if args.single_dtype:
        dtypes = [item for item in DTYPES if item[0] == args.single_dtype]

    print("\nMUSA Multiply smoke benchmark")
    print(f"warmup={args.warmup} iters={args.iters} repeat={args.repeat}")
    print(f"target: >= {HardwareSpec.PASS_RATIO * 100:.0f}% of theoretical bandwidth")
    print_header()

    results = []
    seed = 2026
    for shape_index, (lhs_shape, rhs_shape) in shape_items:
        for dtype_index, dtype_info in enumerate(dtypes):
            result = run_one_case(
                lhs_shape,
                rhs_shape,
                dtype_info,
                args,
                seed + shape_index * 10 + dtype_index,
                shape_index,
            )
            results.append(result)
            print_result(result)

    if args.json:
        save_json(args.json, results)


if __name__ == "__main__":
    main()
