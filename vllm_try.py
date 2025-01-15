from vllm import LLM, SamplingParams

prompts = [
    "The capital of France is",
    "The capital of Ireland is",
    "The capital of Poland is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model="meta-llama/Llama-3.2-1B")

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")