# Experiment 2: Paged KV Cache + BlockMask (Random Tensors, GPU)

Goal: build a minimal correctness harness for paged KV caching with FlexAttention using random tensors on GPU. This validates the core indexing, cache writes, and mask logic without loading any HF model weights.

Why this experiment: it exercises the exact mechanics needed for Flex-Nano-vLLM (page table, logical-to-physical mapping, and BlockMask construction) while staying lightweight and fully controlled.

Prereqs
- GPU available (CUDA).
- PyTorch 2.5+ with `torch.nn.attention.flex_attention`.

Overview
You will:
1) Implement a tiny `PageTable` and `PagedKVCache`.
2) Populate the cache using random K/V tensors at logical positions.
3) Build a BlockMask to isolate each sequence’s pages within a shared cache.
4) Compare FlexAttention output vs a naive attention reference.

Experiment Steps (no code provided)

1) Choose dimensions
- `page_size`: 128 (or 16 for faster debug).
- `n_pages`: total pages in the shared cache (e.g., 16).
- `n_heads`, `head_dim`: small (e.g., 2 heads, 16 dim).
- `batch`: number of sequences (start with 1, then 2).
- `seq_lens`: lengths per sequence (e.g., [20] then [20, 180]).

2) Implement PageTable (minimal)
- `allocate()` returns a free batch index.
- `reserve(batch_idx, seq_len)` allocates enough pages so `capacity >= seq_len`.
- `erase(batch_idx)` frees its pages.
- Store:
  - `page_table[batch, logical_block] -> physical_page`
  - `physical_to_logical[batch, physical_page] -> logical_block`
  - `capacity[batch]`

3) Implement PagedKVCache
- Backing tensors:
  - `k_cache` and `v_cache` shaped `[1, H, N, D]` where `N = n_pages * page_size`.
- `assign(batch_idx, input_pos, k_val, v_val)`:
  - Map `input_pos` -> logical block + offset.
  - Use `page_table` to map logical block -> physical page.
  - Compute absolute cache address: `physical_page * page_size + offset`.
  - Write the K/V into `k_cache` and `v_cache` at those addresses.

4) Populate cache with random tensors
- For each sequence, create random `k_val`, `v_val` of shape `[H, seq_len, D]`.
- Write them into the cache at the correct logical positions using `assign`.
- Keep a per-sequence copy of the original K/V so you can build a reference.

5) Build the BlockMask for multi-sequence decoding
- Query shape: `[B, H, 1, D]` (one new token per sequence).
- Key/Value shape: `[1, H, N, D]` (shared cache).
- Mask rule for each batch element:
  - It can attend only to physical pages assigned to that sequence.
  - It must be causal: only positions <= current logical position.
- Options:
  - Use `create_block_mask(mask_mod, B, H, Q_len=1, KV_len=N)`.
  - `mask_mod(b, h, q_idx, kv_idx)` should:
    - Map `kv_idx` to `physical_page = kv_idx // page_size`.
    - Accept only if `physical_page` is in that sequence’s assigned pages.
    - Enforce causal: `kv_idx <= max_valid_kv_idx_for_sequence`.

6) Reference (naive) attention
- Build a per-sequence contiguous KV tensor from the original random K/V.
- Compute attention for each sequence:
  - Use `scaled_dot_product_attention` or manual softmax with a causal mask.
  - Query is a random tensor (or the next-token query you generate).
- This yields a reference output `[B, H, 1, D]`.

7) Compare outputs
- Run FlexAttention with shared cache + BlockMask.
- Compare with reference per sequence:
  - `max_abs_diff` should be very small (e.g., < 1e-4 in float32).

8) Test cases
- Case A: single sequence, length < page_size.
- Case B: two sequences, lengths in different pages (e.g., 20 and 180).
- Case C: page boundary: length exactly `page_size`, then `page_size + 1`.

Acceptance Criteria
- FlexAttention output matches naive reference within a small tolerance for all cases.
- No out-of-bounds cache writes.
- Mask does not leak attention across sequences.

Optional (Meta-only HF inspection)
- Instantiate a small HF model on `device="meta"` just to inspect config
  and confirm head_dim / num_heads / layout. Do not run forward passes.

Notes
- Keep tensors in float32 for clarity; switch to bfloat16 later.
- Start with small `page_size` (e.g., 16) to make debugging easier.
- Log `page_table`, assigned pages, and `max_valid_kv_idx` when debugging.
