#!/usr/bin/env python3
"""Simple smoke benchmark for the MUSA SelectV2 operator."""

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
    ([], [1024], []),
    ([1024], [1024], [1024]),
    ([1024 * 1024], [1024 * 1024], [1024 * 1024]),
    ([1024, 1024], [1024, 1024], [1024, 1024]),
    ([1024], [1024, 1024], [1024, 1024]),
    ([1024, 1], [1024, 1], [1, 1024]),
    ([2], [2, 1, 2, 1], [2, 1, 2, 1]),
    ([], [1, 2, 2, 2, 1], [1, 2, 2, 1]),
]


def select_output_shape(cond_shape, lhs_shape, rhs_shape):
    return broadcast_shape(cond_shape, broadcast_shape(lhs_shape, rhs_shape))


def make_condition(shape, output_shape, rng):
    if shape:
        return rng.randint(0, 2, size=shape).astype(np.bool_)
    if num_elements(output_shape) % 2 == 0:
        return np.array(False, dtype=np.bool_)
    return np.array(True, dtype=np.bool_)


def build_graph(cond_shape, lhs_shape, rhs_shape, dtype_name, tf_dtype, np_dtype, device, seed):
    rng = np.random.RandomState(seed)
    out_shape = select_output_shape(cond_shape, lhs_shape, rhs_shape)
    cond_np = make_condition(cond_shape, out_shape, rng)
    lhs_np = make_data(lhs_shape, dtype_name, np_dtype, rng)
    rhs_np = make_data(rhs_shape, dtype_name, np_dtype, rng)
    expected = np.where(cond_np, lhs_np, rhs_np)

    graph = tf.Graph()
    with graph.as_default():
        with tf.device(device):
            cond = tf.constant(cond_np, dtype=tf.bool, name="cond")
            lhs = tf.constant(lhs_np, dtype=tf_dtype, name="lhs")
            rhs = tf.constant(rhs_np, dtype=tf_dtype, name="rhs")
            output = tf.where(cond, lhs, rhs, name="select")
            output = tf.identity(output, name="output")
            with tf.control_dependencies([output]):
                step = tf.no_op(name="step")
    return graph, step, output, expected


@profile_with_mcu("select", Path(__file__).resolve())
@attach_perf_metrics
def run_one_case(shape_case, dtype_info, args, seed, shape_index):
    cond_shape, lhs_shape, rhs_shape = shape_case
    dtype_name, tf_dtype, np_dtype, dtype_bytes = dtype_info
    graph, step, output, expected = build_graph(
        cond_shape, lhs_shape, rhs_shape, dtype_name, tf_dtype, np_dtype, args.device, seed
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

    out_shape = select_output_shape(cond_shape, lhs_shape, rhs_shape)
    output_elements = num_elements(out_shape)
    p50_ms = float(np.percentile(times, 50) * 1000.0)
    p90_ms = float(np.percentile(times, 90) * 1000.0)
    min_ms = float(np.min(times) * 1000.0)

    return {
        "dtype": dtype_name,
        "cond_shape": cond_shape,
        "lhs_shape": lhs_shape,
        "rhs_shape": rhs_shape,
        "output_shape": out_shape,
        "output_elements": output_elements,
        "p50_ms": p50_ms,
        "p90_ms": p90_ms,
        "min_ms": min_ms,
        "effective_ops": output_elements,
        "effective_bytes": output_elements * dtype_bytes * 3 + num_elements(cond_shape) * 1,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Simple SelectV2 smoke benchmark.")
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

    print("\nMUSA SelectV2 smoke benchmark")
    print(f"warmup={args.warmup} iters={args.iters} repeat={args.repeat}")
    print(f"target: >= {HardwareSpec.PASS_RATIO * 100:.0f}% of theoretical bandwidth")
    print_header()

    results = []
    seed = 2026
    for shape_index, shape_case in shape_items:
        for dtype_index, dtype_info in enumerate(dtypes):
            result = run_one_case(
                shape_case,
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
