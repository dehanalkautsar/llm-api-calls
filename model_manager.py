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
            inputs_path = os.path.join(package_dir, f"{config['project_name']}/inputs/{id_prefix}.jsonl")
            with open(inputs_path, "w+") as f:
                for item in requests:
                    json.dump(item, f)
                    f.write('\n')

            # delete and replace if there already exists
            for uploaded_files in self.model.files.list():
                if uploaded_files.filename == f"{id_prefix}.jsonl":
                    print(f"{id_prefix}.jsonl already exists, replacing...")
                    self.model.files.delete(uploaded_files.id)
                    break
            
            batch_input_file = self.model.files.create(
                file=open(inputs_path, "rb"),
                purpose="batch"
            )

            batch_obj = self.model.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                    "project_name" : config['project_name']
                }
            )

            print("batches created!")
            # TODO : SAVE BATCH NUMBERS IN A FILE
    
    def check_and_download_batches(self, batch_numbers, package_name=None):
        for bn in batch_numbers:
            batch = self.model.batches.retrieve(bn)
            if batch.status == "completed":
                file_content = self.model.files.content(batch.output_file_id)
                
                if(package_name):
                    base_output_path = f"{package_name}/outputs"
                else:
                    base_output_path = "outputs"
                os.makedirs(base_output_path, exist_ok=True)

                # save raw output
                with open(f"{base_output_path}/raw_results.jsonl", "wb") as f:
                    f.write(file_content.content)

                # quickview save
                quickview = {}
                dct_strings = file_content.content.decode('utf-8').strip().split("\n")
                for dct_string in dct_strings:
                    data = json.loads(dct_string)
                    prefix = self.get_prefix_from_id(data["custom_id"])
                    if prefix in quickview:
                        quickview[prefix].append({
                            "id" : data["custom_id"].split("_")[-1],
                            "content" : data["response"]["body"]["choices"][0]["message"]["content"],
                        })
                    else:
                        quickview[prefix] = [{
                            "id" : data["custom_id"].split("_")[-1],
                            "content" : data["response"]["body"]["choices"][0]["message"]["content"],
                        }]

                with open(f"{base_output_path}/quick_results.json", "w") as qf:
                    json.dump(quickview, qf, indent=4)

    @staticmethod
    def get_prefix_from_id(id_string):
        spl = id_string.split("_")
        return "_".join(spl[:-1])

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