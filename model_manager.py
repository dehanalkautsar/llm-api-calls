import os, json
from typing import List, Dict, Any, Union
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

OPENAI_MODELS = ["gpt-4o-mini"]

class ModelManager:
    def run_grid(self, grid: List[Dict[Any, Any]]):
        in_one_context_idxes = []
        
        # check if inputs (only the i1c's) are valid
        for job in grid["data"]:
            if job["config"]["in_one_context"]:
                self._is_i1c_input_shape_valid(job["input_prompts"])
        
        # submit all first
        for jobidx, job in enumerate(grid["data"]):
            if job["config"]["in_one_context"]:
                in_one_context_idxes.append(jobidx)
            agent = self._instantiate_model_based_on_name(job["config"]["model_name"])
            agent(grid["grid_info"],
                  job["config"],
                  job["input_prompts"])
        print("grid finished submitting/running. continuing with in one context prompts...")

        if len(in_one_context_idxes) > 0:
            print("in one context prompts detected, running...")
            for i in in_one_context_idxes:
                print(f"running ... {i}")

    def _instantiate_model_based_on_name(self, model_name):
        if model_name in OPENAI_MODELS:
            return CustomOpenAIAgent()
        else:
            raise ValueError(f"Invalid or unregistered model name! {model_name}")
        
    def _is_i1c_input_shape_valid(self, i1c_inputs):
        if(type(i1c_inputs[0]) == list):
            first_length = len(i1c_inputs[0])
            for inner in i1c_inputs:
                if len(inner) != first_length:
                    raise ValueError("len must be the same for each chat window")
            return True
        else:
            raise ValueError("i1c_inputs must be a 2D list of strings")

class CustomOpenAIAgent():
    def __init__(self):
        self.model = OpenAI()
    
    def __call__(self,
                 grid_info,
                 config,
                 input_prompts,
                 history = None) -> Union[str, List[str]]:
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

        # constructing the requests
        requests = []
        
        if(config["in_one_context"]):
            CURRENT_IDX = 1
            input_prompts = [row[CURRENT_IDX] for row in input_prompts]

        print("INPUT PROMPTS")
        print(input_prompts)

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

        input_dir_name = os.path.join(os.path.dirname(__file__),
                                    f"{grid_info['project_name']}/inputs/{grid_info['run_name']}/")
        just_file_name = f"{grid_info['run_name']}_{id_prefix}.jsonl"
        inputs_path = os.path.join(input_dir_name, just_file_name)
        os.makedirs(os.path.dirname(inputs_path), exist_ok=True)
        with open(inputs_path, "w+") as f:
            for item in requests:
                json.dump(item, f)
                f.write('\n')

        # # delete and replace if there already exists
        # for uploaded_files in self.model.files.list():
        #     if uploaded_files.filename == just_file_name:
        #         print(f"{just_file_name} already exists, replacing...")
        #         self.model.files.delete(uploaded_files.id)
        #         break
        
        # batch_input_file = self.model.files.create(
        #     file=open(inputs_path, "rb"),
        #     purpose="batch"
        # )

        # batch_obj = self.model.batches.create(
        #     input_file_id=batch_input_file.id,
        #     endpoint="/v1/chat/completions",
        #     completion_window="24h",
        #     metadata={
        #         "project_name" : grid_info['project_name'],
        #         "run_name" : grid_info['run_name']
        #     }
        # )
        
        batch_obj_id = f"this is a mock id from {"i1c" if config["in_one_context"] else "wle"}"

        print(f"batch created! batch id : {batch_obj_id}", end=" ---- ")
        # print(f"batch created! batch id : {batch_obj.id}", end=" ---- ")

        if(config["in_one_context"]):
            batchid_path = f"{grid_info['project_name']}/outputs/{grid_info['run_name']}/i1c_batch_ids.txt"
        else:
            batchid_path = f"{grid_info['project_name']}/outputs/{grid_info['run_name']}/batch_ids.txt"
        
        print(f"saving in: {batchid_path}")
        batchid_file = os.path.join(os.path.dirname(__file__), batchid_path)
        os.makedirs(os.path.dirname(batchid_file), exist_ok=True)

        with open(batchid_file, "a") as bidf:
            bidf.write(batch_obj_id + "\n")
            # bidf.write(batch_obj.id + "\n")
    
    def check_and_download_batches(self, project_name, run_name, batch_numbers):
        raw_results = []
        quickview = {}
        for bn in batch_numbers:
            batch = self.model.batches.retrieve(bn)
            print(f"batch_status {bn}", batch.status)
            if batch.status == "completed":
                file_content = self.model.files.content(batch.output_file_id)
                file_content_string = file_content.content.decode('utf-8')
                raw_results.append(file_content_string)
                dct_strings = file_content_string.strip().split("\n")
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
            else:
                print(f"batch not done. batch status {batch.status}\nREMINDER: ye might need to rerun the check_results script to get the full results")
        
        base_output_path = os.path.join(os.path.dirname(__file__), f"{project_name}/outputs/{run_name}")

        # save raw
        with open(f"{base_output_path}/raw_results.jsonl", "w") as f:
            f.write("".join(raw_results))
        
        # save quick results
        with open(f"{base_output_path}/quick_results.json", "w") as qf:
            json.dump(quickview, qf, indent=4)

    @staticmethod
    def get_prefix_from_id(id_string):
        spl = id_string.split("_")
        return "_".join(spl[:-1])
