from modeling_qwen3_eager_att import Qwen3ForCausalLM
from modeling_qwen3_flex_att import Qwen3ForCausalLM as Qwen3ForCausalLMFlex
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache
import torch
import time


device = "cuda"
model_path = "/home/thiago/repos/LearningFlexNanovLLM/experiments/AddingFlexAttentionQwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_path)


def greedy_decode_50_steps(model, input_ids):
    model.eval()
    max_new_tokens = 50
    generated = input_ids
    past_key_values = DynamicCache()
    with torch.no_grad():
        # First pass over the full prompt, then take 1 token
        outputs = model(
            input_ids=generated,
            use_cache=True,
            past_key_values=past_key_values,
            logits_to_keep=1,
        )
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)

        # Remaining 49 steps, feeding only the last token each time
        for _ in range(max_new_tokens - 1):
            outputs = model(
                input_ids=next_token,
                use_cache=True,
                past_key_values=past_key_values,
                logits_to_keep=1,
            )
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)

    return generated


# Eager attention model
model = Qwen3ForCausalLM.from_pretrained(model_path, device_map=device)
prompt = "Hello, how are you?"
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"].to(device)

start_time = time.time()
gen_ids = greedy_decode_50_steps(model, input_ids)
end_time = time.time()
output_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
print(f"Time taken (eager): {end_time - start_time} seconds")


# Flex attention model
model_flex = Qwen3ForCausalLMFlex.from_pretrained(model_path, device_map=device,torch_dtype=torch.bfloat16)
start_time = time.time()
gen_ids_flex = greedy_decode_50_steps(model_flex, input_ids)
end_time = time.time()
output_text_flex = tokenizer.decode(gen_ids_flex[0], skip_special_tokens=True)
print(f"Time taken (flex): {end_time - start_time} seconds")
