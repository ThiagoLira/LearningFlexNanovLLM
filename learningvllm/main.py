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

