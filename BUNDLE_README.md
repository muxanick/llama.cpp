# `claude/dflash-optimization-bundle` — LLM-discovered CUDA optimizations

> **Disclosure.** This branch was prepared end-to-end by Claude (Anthropic
> LLM, model `claude-opus-4-7`) acting under my direction within an iterative
> debugging + optimization workflow on a private downstream project (an
> experimental Qwen3.5/3.6 spec-decode inference daemon). The optimizations
> were discovered, implemented, tested, and benchmarked by the LLM; I
> reviewed and pushed. I am sharing this branch publicly as a
> **demonstration** that LLMs can do this caliber of CUDA-backend work, and
> to surface the techniques in case any of them are useful to other
> implementers. I am **not** opening a pull request — per
> [AGENTS.md](AGENTS.md), llama.cpp does not accept fully AI-generated PRs,
> and I want to respect that. The branch + this README are the artifact.

## What's in the bundle

Five CUDA-backend optimizations on top of `upstream/master`
(`d14ce3dab` post-MTP-merge, b9235). Env-gated where they change
behavior; the K-quant kernels and the CUDA FWHT kernel are pure
additions that only engage when the corresponding code paths are
exercised.

| Component | Files | Behavior change |
|---|---|---|
| **c1** K-quant `get_rows` kernels (Q2_K..Q6_K) | `ggml-cuda/getrows.cu`, `ggml-cuda/ggml-cuda.cu` (supports_op) | Always on — fills a gap (upstream rejected K-quant `get_rows` as unsupported pre-this-branch) |
| **c2** `ggml_cuda_graph_update_required` skip-padding | `ggml-cuda/ggml-cuda.cu` (~line 3287) | Env-gated `GGML_CUDA_PROPS_SKIP_PADDING=1` |
| **c3** FA fixed-parallel-blocks env-gate | `ggml-cuda/fattn-common.cuh` (~line 916+) | Env-gated `GGML_FATTN_FIXED_PARALLEL_BLOCKS=N`, scoped to `ncols==1` |
| **c4** CUDA FWHT for `GGML_HINT_SRC0_IS_HADAMARD` | `ggml-cuda/fwht.cu`, `ggml-cuda/fwht.cuh`, `ggml-cuda/ggml-cuda.cu` (hint dispatch) | Always on — closes the CUDA gap left by merged PR #22631 (CPU FWHT) |
| **c6** Explicit graph-key hint on `ggml_backend_cuda_context` | `ggml-cuda/common.cuh`, `ggml-cuda/ggml-cuda.cu` | Always plumbed; default (null hint) is byte-identical to master |

## Why each one exists (the diagnostic story)

### c1 — K-quant `get_rows` kernels

Discovered while profiling a Qwen3.6-35B-A3B-IQ2_XXS workload. Upstream
`get_rows` supports F16/F32/BF16/I32/Q1_0/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 but
hits the `default:` arm with `GGML_ABORT` for K-quants. K-quant token
embeddings (e.g. Q4_K `token_embd.weight` on Qwen3.6) therefore had to
be dequantized on CPU and DMA'd to GPU each step — costing ~1.4 ms/tok
of H2D bandwidth on the small-batch decode path.

Adding native CUDA `get_rows` kernels for Q2_K..Q6_K eliminates that
H2D entirely. The kernels mirror the corresponding
`dequantize_block_q*_K` math from `convert.cu` but operate one output
element per thread (matching the launch geometry of
`get_rows_cuda_float`) rather than the per-block layout of the bulk
dequant path.

**Validation:** `test-backend-ops -o GET_ROWS` adds 20 new test cases
(5 K-quant types × 4 shape configs), all PASS byte-identically to the
CPU reference.

### c2 — `ggml_cuda_graph_update_required` skip-padding

Discovered while diagnosing a "warmup reset" loop on a sched-routed
prefill workload. Under CUDA graphs (`GGML_CUDA_GRAPHS=1`), every
prefill chunk produced ~59 spurious graph-reset events per inference
even though the cgraph shapes were stable.

Root cause: `properties_changed` uses
`memcmp(&graph->node_props[i], &prop, sizeof(prop))` to detect changes
between rebuilds, and `prop` embeds a full `ggml_tensor` struct. The
struct has **uninitialized padding bytes between `enum ggml_type type`
(offset 0–3) and the next pointer-aligned field (offset 8–15)**. The
CUDA allocator does not zero those padding bytes — they carry whatever
garbage was last there. Two byte-identical-content tensors can have
different padding bytes if they came from different allocations, and
the `memcmp` flags them as different → graph reset → recapture cost.

The fix replaces the whole-struct `memcmp` with a field-by-field
compare that skips the padding (and intentionally also skips `name`
and `extra` — see comment in code). Captured live on the same
workload: 59 → 0 spurious resets, ~0.5 % wall savings on the
benchmarked trace.

**Env-gated** so behavior is unchanged unless `GGML_CUDA_PROPS_SKIP_PADDING=1`
is set, in case the relaxation is overzealous on workloads I didn't
test.

### c3 — FA fixed-parallel-blocks env-gate

Discovered while trying to keep CUDA graphs alive across KV-cache
bucket transitions on AR decode. The FA kernel's `gridDim.y`
(`parallel_blocks`) is computed dynamically from `K->ne[1]` —
specifically, as the result of an efficiency-search loop that picks
the best wave-occupancy `parallel_blocks` value. When `K->ne[1]`
changes between iterations (which it does on AR decode as the KV grows
or, more importantly, when a bucketed FA-read window changes
buckets), `parallel_blocks` changes too. `cudaGraphExecUpdate` can
patch tensor pointers and runtime args, but it cannot change
`gridDim.y`, so a graph captured at one `parallel_blocks` value
cannot be reused at another.

The env-gate decouples them. When `GGML_FATTN_FIXED_PARALLEL_BLOCKS=N`
is set, FA locks `parallel_blocks=N` regardless of `K->ne[1]`.

**Scoping is critical.** I scoped this to `ncols == 1` (vec FA /
AR-decode kernel). The prefill kernels (`ncols > 1`, `mma-f16`,
`tile`) share `parallel_blocks` with the vec kernel but lack the
gridDim.y-stride safety that lets excess blocks exit cleanly. An
unscoped patch broke prefill correctness in testing — the scoped
patch preserves prefill byte-equality. There is a `(ncols == 1)`
runtime check (cheap, compiler folds it away for kernels where ncols
is fixed at instantiation).

The combine kernel (`flash_attn_combine_results`) already handles
excess blocks correctly: blocks beyond `ntiles_KV` produce zero-meta
contributions, and the `expf(meta.x - kqmax) ~= 0` masking zeroes
them out.

### c4 — CUDA FWHT for `GGML_HINT_SRC0_IS_HADAMARD`

PR #22631 (merged 2026-05-05) shipped a CPU implementation of the Fast
Walsh-Hadamard Transform that runs when `ggml_mul_mat_set_hint(t,
GGML_HINT_SRC0_IS_HADAMARD)` is set on a `MUL_MAT` node — replacing the
$O(N^2)$ literal matmul against a Hadamard matrix with $O(N \log N)$
FWHT on `src1`. The hint is set on two paths in `src/llama-graph.cpp`
and `src/llama-kv-cache.cpp`. On the CPU backend the savings are real;
on the CUDA backend, the hint was silently ignored and `ggml_cuda_mul_mat`
fell through to a literal matmul.

This component adds the CUDA-side dispatch hook in `ggml_cuda_mul_mat`
(checks `op_params[1]` against `GGML_HINT_SRC0_IS_HADAMARD`, routes to
the new `ggml_cuda_op_fwht` when set) and a CUDA kernel that mirrors
the CPU algorithm: one CUDA block per row, threads cooperate via
shared memory across `log2(n)` butterfly passes after a `1/sqrt(n)`
scale. Block size is 128 (4 warps); shared-memory budget caps `n` at
1024 elements per row, which is well above any Hadamard-using path in
practice (the upstream callers use `head_dim`, typically 64-128).

**Numeric note.** Both the pre- and post-c4 CUDA paths produce
mathematically equivalent output — the pre-c4 path was correct (literal
$H \cdot x$ with $H = $ scaled Hadamard equals $\mathrm{FWHT}(x)$) but
slow ($O(n^2)$). c4 brings the algorithmic class to parity with CPU.

**Validation.** `test-backend-ops -o MUL_MAT_HADAMARD` enables the 4
existing FWHT test cases (n = 64, 128, 256; batch shapes 1, 32) for
the CUDA backend; all pass byte-identically to the CPU reference.

### c6 — Explicit graph-key hint

Discovered while building a bucket-aware speculative-decode loop where
each step builds a fresh `ggml_cgraph` (the cgraph ctx is freed and
re-initialized between iterations). `ggml_cuda_graph_get_key` uses
`cgraph->nodes[0]` as the cache key — but `nodes[0]` is the address
of a freshly-allocated tensor, so it changes every iteration even
though the logical computation is identical. Result: every iteration
falls into "first-time-seen" path, the cuda graph is rebuilt from
scratch, and `cudaGraphExecUpdate` never gets exercised.

The hint API lets the caller supply a stable key. The hint is a
`std::atomic<const void *>` on `ggml_backend_cuda_context`. Before
invoking a compute pass, the caller sets `cuda_ctx->next_graph_key_hint`
to whatever stable identifier they want (e.g. a per-bucket sentinel).
`ggml_cuda_graph_get_key` loads the hint with relaxed ordering and
falls back to `cgraph->nodes[0]` if null. The hint is cleared at the
end of `ggml_backend_cuda_graph_compute` so it cannot leak to the
next call.

When no caller sets the hint, behavior is byte-identical to master
(load returns null, fallback path runs). This is the only component
that adds public-ish API surface (the atomic field is reachable from
anywhere that has a `ggml_backend_cuda_context *`), and I acknowledge
it's the most "dflash-coupled" — it exists to serve a downstream
caller's bucketing scheme rather than a generic upstream need. I left
it in the bundle because (a) the absence of any external-key
mechanism is itself a documented pain point in CUDA-graph-using
projects, and (b) the diff is small enough to assess on its own.

## How to verify

Both the K-quant kernel correctness (c1) and the CUDA-graph
behavior (c2/c3/c6) are verifiable on a standard build:

```
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release --target test-backend-ops llama-cli llama-bench --parallel 8
./build/bin/Release/test-backend-ops -o GET_ROWS        # c1 — adds 20 PASS lines for q2_K..q6_K
GGML_CUDA_PROPS_SKIP_PADDING=1 ./build/bin/Release/llama-cli ...  # c2 — observe reset count drop
GGML_FATTN_FIXED_PARALLEL_BLOCKS=4 ./build/bin/Release/llama-cli ...  # c3 — observe graph reuse across buckets
```

## Reference baseline (this branch's host)

Captured on RTX 4090 Laptop (16 GB VRAM), CUDA 13.2, sm_89, MSVC
19.44.

| Test | Configuration | tok/s |
|---|---|---|
| `llama-bench pp32` | Unsloth `Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf`, `--n-cpu-moe 32 -ngl 99` | 103.74 |
| `llama-bench tg32` | same | 49.57 |
| `llama-cli --spec-type draft-mtp --spec-draft-n-max 2` | same model, `-ngl 36` (4 layers CPU) | 37.3 (gen) |

These are baseline numbers from `upstream/master` (without the
bundle); the bundle's net effect on this exact hardware is left to
the reader's measurement, since the components serve different code
paths and only matter when those paths are exercised.

## What's deliberately NOT in this bundle

- **Multi-Token Prediction.** Already merged upstream via PR #22673
  (am17an, 2026-05-16). Use `--spec-type draft-mtp` against Unsloth's
  `Qwen3.6-35B-A3B-MTP-GGUF` or `Qwen3.6-27B-MTP-GGUF`.
- **CPU FWHT for KV cache rotation.** Already merged upstream via PR
  #22631 (AlrIsmail, 2026-05-05). The matching CUDA kernel is now in
  this bundle as c4 above.
- **TurboQuant KV cache types.** See ongoing PR #21089 (elusznik).
- **MoE expert-sum fusion / per-bucket cuda_graph map / `set_rows`
  fusion patterns** — these are tightly coupled to the downstream
  project's graph builder and unlikely to upstream cleanly.

## Companion: Unsloth Qwen3.6-MTP recommended params

For benching against the Unsloth MTP GGUFs the project at
[https://unsloth.ai/docs/models/qwen3.6#mtp-guide](https://unsloth.ai/docs/models/qwen3.6#mtp-guide)
recommends `--spec-draft-n-max 2` (n=2 = 83% accept; n=4 drops to
50%) and per-mode sampling defaults. Use those for apples-to-apples.
