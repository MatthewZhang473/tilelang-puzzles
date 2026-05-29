"""
Puzzle 03: Outer Vector Add
==============
In this puzzle we will enter the 2D world!

Category: ["official"]
Difficulty: ["easy"]
"""

import tilelang
import tilelang.language as T
import torch
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.utils import test_puzzle

"""
Consider an outer vector addition operation. The result is a matrix where
each element (i, j) is the sum of A[i] and B[j].

The main difference from the previous puzzle is that C is now a 2D tensor and
we have two different iterators in buffers A and B. So the dataflow is also
a little different.

But remember that any N dimensional tensor can be viewed as a 1D tensor in memory.
So we just need to handle the indexing properly.

03-1: Outer vector addition.

Inputs:
    A: Tensor([N,], float16)  # input tensor
    B: Tensor([M,], float16)  # input tensor
    N: int   # size of the tensor. 1 <= N <= 8192
    M: int   # size of the tensor. 1 <= M <= 8192

Output:
    C: [N, M]  # output tensor

Definition:
    for i in range(N):
        for j in range(M):
            C[i, j] = A[i] + B[j]
"""


def ref_outer_add(A: torch.Tensor, B: torch.Tensor):
    assert len(A.shape) == 1
    assert len(B.shape) == 1
    assert A.dtype == B.dtype == torch.float16
    return torch.add(input=A[:, None], other=B[None, :])


@tilelang.jit
def tl_outer_add(A, B, BLOCK_N: int, BLOCK_M: int):
    N, M = T.const("N, M")
    dtype = T.float16
    A: T.Tensor((N,), dtype)
    B: T.Tensor((M,), dtype)
    C = T.empty((N, M), dtype)

    # TODO: Implement this function

    num_blocks_N = N // BLOCK_N
    num_blocks_M = M // BLOCK_M

    with T.Kernel(num_blocks_N * num_blocks_M, threads=256) as bx:
        bx_N = bx // num_blocks_M  # floor division
        bx_M = bx % num_blocks_M
        A_local = T.alloc_fragment((BLOCK_N,), dtype=dtype)
        B_local = T.alloc_fragment((BLOCK_M,), dtype=dtype)
        C_local = T.alloc_fragment((BLOCK_N, BLOCK_M), dtype=dtype)
        T.copy(A[bx_N * BLOCK_N : (bx_N + 1) * BLOCK_N], A_local)
        T.copy(B[bx_M * BLOCK_M : (bx_M + 1) * BLOCK_M], B_local)

        for i in T.Parallel(BLOCK_N):
            for j in T.Parallel(BLOCK_M):
                C_local[i, j] = A_local[i] + B_local[j]
        T.copy(
            C_local,
            C[
                bx_N * BLOCK_N : (bx_N + 1) * BLOCK_N,
                bx_M * BLOCK_M : (bx_M + 1) * BLOCK_M,
            ],
        )

    return C


def run_outer_add():
    print("\n=== Outer Vector Add ===\n")
    N = 8192
    M = 4096
    BLOCK_N = 1024
    BLOCK_M = 1024

    tl_outer_add_kernel = tl_outer_add.compile(
        N=N, M=M, BLOCK_N=BLOCK_N, BLOCK_M=BLOCK_M
    )
    tl_outer_add_kernel.export_sources(kernel_path="./tmp/03-puzzle-kernel.cu")

    test_puzzle(
        tl_outer_add,
        ref_outer_add,
        {"N": N, "M": M, "BLOCK_N": BLOCK_N, "BLOCK_M": BLOCK_M},
    )


if __name__ == "__main__":
    run_outer_add()
