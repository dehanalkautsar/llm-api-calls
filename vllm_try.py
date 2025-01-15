from vllm import LLM, SamplingParams

prompts = [
    [
        {
            "role" : "user",
            "content" : "What is the capital of Ireland"
        },
        {
            "role" : "assistant",
            "content" : "Dublin"
        },
        {
            "role" : "user",
            "content" : "What is the currency?"
        },
    ],
        [
        {
            "role" : "user",
            "content" : "What is the capital of Japan"
        },
        {
            "role" : "assistant",
            "content" : "Tokyo"
        },
        {
            "role" : "user",
            "content" : "What is the currency?"
        },
    ],
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model="meta-llama/Llama-3.2-1B")

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")