import os, json
from typing import List, Dict, Any, Union
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

OPENAI_MODELS = ["gpt-4o-mini"]

class CustomOpenAIAgent():
    def __init__(self):
        self.model = OpenAI()
    
    def __call__(self, config, input_prompts) -> Union[str, List[str]]:
        """
        calls openai with the config provided. defaults to async calls (except when it's on the same window)
        setting n_concurrent = 1 is just single calls
        """
        in1ctx = "1ctx" if config["in_one_context"] else "sep"
        if("args" in config):
            arg_string = "-".join([f"{k}_{v}" for k, v in config["args"].items()])
            id_prefix = f"{config["model_name"]}-{in1ctx}-{arg_string}"
        else:
            id_prefix = f"{config["model_name"]}-{in1ctx}"

        if config["in_one_context"]:
            print("yea do in one")
        else:
            requests = []
            for idx, prompt in enumerate(input_prompts):
                requests.append({
                    "custom_id": id_prefix + f"_id{str(idx+1)}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": config["model_name"],
                        "messages": [{
                            "role": "user",
                            "content": prompt
                        }],
                        **config.get("args", {})
                    }
                })

            package_dir = os.path.dirname(__file__)
            with open(os.path.join(package_dir, f"{config['project_name']}/inputs/{id_prefix}.jsonl"), "w+") as f:
                for item in requests:
                    json.dump(item, f)
                    f.write('\n')

            # {
            #     "custom_id": "request-1",
            #     "method": "POST",
            #     "url": "/v1/chat/completions",
            #     "body": {
            #         "model": "gpt-3.5-turbo-0125",
            #         "messages": [
            #         {
            #             "role": "system",
            #             "content": "You are a helpful assistant."
            #         },
            #         {
            #             "role": "user",
            #             "content": "Hello world!"
            #         }
            #         ],
            #         "max_tokens": 1000
            #     }
            # }

        # chat_completion = client.chat.completions.create(
        #     messages=[
        #         {
        #             "role": "user",
        #             "content": "What is the capital of indonesia?",
        #         }
        #     ],
        #     model="gpt-4o-mini",
            
        # )

        # print(chat_completion)

class ModelManager:
    def _instantiate_model_based_on_name(self, model_name):
        if model_name in OPENAI_MODELS:
            return CustomOpenAIAgent()
        else:
            raise ValueError(f"Invalid or unregistered model name! {model_name}")
        
    def run_grid(self, grid: List[Dict[Any, Any]]):
        for job in grid:
            agent = self._instantiate_model_based_on_name(job["config"]["model_name"])
            agent(job["config"], job["input_prompts"])

# grid example:
# [
#     {
#         model_name : "model_name",
#         in_one_context : True,
#         args : {
#             temperature : 0,
#             top_p : 0,
#         }
#         prompts : [
#             "hello this is prompt one",
#             "hello this is prompt one",
#             "hello this is prompt one",
#         ]
#     },
#     {
#         model_name : "model_name2",
#         in_one_context : True,
#         args : {
#             temperature : 0,
#             top_p : 0,
#         }
#         prompts : [
#             "hello this is prompt one",
#             "hello this is prompt one",
#             "hello this is prompt one",
#         ]
#     },
# ]



