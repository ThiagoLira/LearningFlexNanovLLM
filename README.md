# Flex-Nano-vLLM — Mini Textbook & Implementation Guide

This is a step-by-step learning notebook for the [flex-nano-vLLM](https://github.com/changjonathanc/flex-nano-vllm) repo, aimed at **advanced Python developers** who want to implement an inference engine from scratch.


Before implementing Flex-Nano-vLLM, this repo will provide simple experiments so one can familiazire with the techniques and modeling involved.

### Experiment 1:

`experiments/AddFlexAttentionQwen`

Adding FlexAttention to HF's implementation of qwen3_modelling and benchmark speed gains.


---

## 1. Overview

Flex-Nano-vLLM implements:
- **Paged KV cache** (memory pages shared across sequences)
- **Dynamic batching** (mix prefill + decode steps)
- **Preemption** (evict low-priority sequences to free memory)
- **FlexAttention** + **BlockMask** (efficient block-sparse attention)

You provide a **preloaded HuggingFace model** (`AutoModelForCausalLM`) and a tokenizer.  
The engine patches the model’s attention layers to use the paged cache.

---

## 2. Class hierarchy

```mermaid
classDiagram
    direction LR

    class Inference {
      +generate(requests)
      -run_one_step()
      -prefill_sequences(batch)
      -get_decoding_block_mask(batch_idx)
      -decode_step(batch_idx, input_ids, input_pos)
      -_decode_step(batch_idx, input_ids, input_pos)
    }

    class PageTable {
      +allocate() -> int
      +reserve(batch_idx, seq_len) -> bool
      +assign(...)
      +assign_prefill_no_paging(...)
      +convert_logical_block_mask(mask, batch_idx)
      +erase(batch_idx)
    }

    class PagedKVCache {
      +update(input_pos, k,v, batch_idx)
    }

    class HFModel~PreTrainedModel~ {
      +model
      +lm_head
    }

    class Gemma2Attention {
      +kv_cache: PagedKVCache
      +forward(...)
    }

    class FlexAttention {
      +flex_attention(...)
      +BlockMask
    }

    Inference --> PageTable : owns
    Inference --> HFModel : uses
    Inference --> FlexAttention : passes masks/opts
    PageTable --> FlexAttention : builds/rewrites BlockMask
    Gemma2Attention --> PagedKVCache : kv_cache.update(...)
    HFModel --> Gemma2Attention : per-layer attention
    PagedKVCache --> PageTable : address mapping
```

3. High-level scheduling flow
   
```mermaid
flowchart TD

A[generate(requests)] --> B[tokenize -> Sequence objects]
B --> C{waiting or running?}
C -->|no| Z[return results]
C -->|yes| D[run_one_step]

D --> E{Can prefill new sequences?}
E -->|yes| F[allocate batch_idx]
F --> G[reserve pages for prompt]
G --> H[build prefill BlockMask]
H --> I[model.forward with flex_attn_*]
I --> J[KV cache updated at each layer]
J --> K[sample one token per sequence]
K --> L[check_done; free pages if finished]
L --> N[move unfinished to running]
N --> C

E -->|no| P[decode step for running sequences]
P --> Q[ensure capacity; reserve if needed]
Q --> R{Out of pages?}
R -->|yes| S[preempt newest running seq -> waiting; erase pages]
R -->|no| T[form batch_idx, input_ids, input_pos]
S --> T
T --> U[get logical decode BlockMask]
U --> V[convert to physical BlockMask]
V --> W[model.forward with flex_attn_*]
W --> X[sample token; check_done]
X --> C
```
4. One decode step (per layer)
```mermaid
sequenceDiagram
    participant Inf as Inference
    participant PT as PageTable
    participant Attn as Gemma2Attention
    participant KVC as PagedKVCache
    participant Flex as FlexAttention

    Inf->>PT: get_decoding_block_mask(batch_idx)
    PT-->>Inf: BlockMask(logical), input_pos
    Inf->>PT: convert_logical_block_mask(mask, batch_idx)
    PT-->>Inf: BlockMask(physical)

    loop each layer
      Inf->>Attn: forward(..., flex_attn_block_mask, flex_attn_input_pos, flex_attn_batch_idx)
      Attn->>KVC: kv_cache.update(...)
      KVC->>PT: assign(...)  
      Attn->>Flex: flex_attention(q, Kcache, Vcache, block_mask)
      Flex-->>Attn: context
      Attn-->>Inf: hidden_states
    end

    Inf->>HF Head: lm_head(last_hidden_state)
    HF Head-->>Inf: logits
```
5. Core components
PageTable

    Allocates/frees pages (allocate, erase)

    Maps logical (batch, block) to physical page index

    Translates BlockMasks from logical to physical space

PagedKVCache

    Holds big [1, H, N, D] k_cache / v_cache

    Writes keys/values via PageTable.assign calls

BlockMask

    Logical: per-sequence causal pattern

    Physical: remapped to unified cache

    mask_mod ensures no cross-sequence attention

6. Homework challenges

    1. Prefix sharing
    Extend PageTable to let two sequences point to same pages until they diverge.

    2. Sliding window
    Implement eviction of oldest pages to keep max length constant.

    3. Prefill+decode fusion
    Overlap prefill of new requests with decode of running ones.

7. How to run with a model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from inference import Inference

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b-it",
    torch_dtype="bfloat16",
    device_map="cuda"
)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")

engine = Inference(
    model=model,
    tokenizer=tokenizer,
    max_batch_size=16,
    max_seq_length=2048,
    n_pages=256,
    page_size=128
)

results = engine.generate(["Hello world!", "Explain paged attention."])
```
