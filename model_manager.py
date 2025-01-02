import os, json
from typing import List, Dict, Any, Union
from dotenv import load_dotenv
from openai import OpenAI
from time import sleep
load_dotenv()

OPENAI_MODELS = ["gpt-4o-mini"]
MAX_CHECK_TRIES = 5

class ModelManager:
    def run_grid(self, grid: List[Dict[Any, Any]]):
        in_one_context_idxes = []
        
        # check if inputs (only the ioc's) are valid
        for job in grid["data"]:
            if job["config"]["in_one_context"]:
                self._is_ioc_input_shape_valid(job["input_prompts"])
        
        # submit all first
        for jobidx, job in enumerate(grid["data"]):
            if job["config"]["in_one_context"]:
                in_one_context_idxes.append(jobidx)
            agent = self._instantiate_model_based_on_name(job["config"]["model_name"])
            agent.run_batch(grid["grid_info"],
                            job["config"],
                            job["input_prompts"])
        print("grid finished submitting/running. continuing with in one context prompts...")
        
        if len(in_one_context_idxes) > 0:
            longest_ioc = max([len(grid["data"][i]["input_prompts"][0]) for i in in_one_context_idxes])
            print("in one context prompts detected, running...")

            for i in range(longest_ioc):
                batchid_path = f"{grid['grid_info']['project_name']}/outputs/{grid['grid_info']['run_name']}/ioc/ioc_batch_ids_idx{i}.txt"
                batchid_file = os.path.join(os.path.dirname(__file__), batchid_path)

                with open(batchid_file, "r") as bf:
                    batchids = [line.strip() for line in bf]
                print(batchids)

                tries = 0
                while(not agent.is_list_of_batch_all_done(batchids) and tries < MAX_CHECK_TRIES):
                    print("waiting for all the ioc batches to finish...")
                    tries += 1
                    sleep(100)

                # batch is done, extract and run next one
                # extract
                agent.extract_batch_results(
                    grid['grid_info']['project_name'],
                    grid['grid_info']['run_name'],
                    batchids
                )

                # run
                for j in in_one_context_idxes:
                    if len(grid["data"][j]["input_prompts"][0]) <= longest_ioc:
                        agent.run_batch(grid["grid_info"],
                                        grid["data"][j]["config"],
                                        grid["data"][j]["input_prompts"],
                                        i+1)

    def _instantiate_model_based_on_name(self, model_name):
        if model_name in OPENAI_MODELS:
            return CustomOpenAIAgent()
        else:
            raise ValueError(f"Invalid or unregistered model name! {model_name}")
        
    def _is_ioc_input_shape_valid(self, ioc_inputs):
        if(type(ioc_inputs[0]) == list):
            first_length = len(ioc_inputs[0])
            for inner in ioc_inputs:
                if len(inner) != first_length:
                    raise ValueError("len must be the same for each chat window")
            return True
        else:
            raise ValueError("ioc_inputs must be a 2D list of strings")

class CustomOpenAIAgent():
    def __init__(self):
        self.model = OpenAI()
    
    def run_batch(self,
                 grid_info,
                 config,
                 input_prompts,
                 ioc_idx = 0) -> Union[str, List[str]]:

        # naming, paths, such and such =======================================================
        in1ctx = "ioc" if config["in_one_context"] else "sep"
        if("args" in config):
            arg_string = "-".join([f"{k}_{v}" for k, v in config["args"].items()])
            batch_name = f"{config["model_name"].replace("-", "_")}-{in1ctx}-{arg_string}"
        else:
            batch_name = f"{config["model_name"].replace("-", "_")}-{in1ctx}"

        dirname = os.path.dirname(__file__)
        if(config["in_one_context"]):
            input_dir_name = os.path.join(dirname, f"{grid_info['project_name']}/inputs/{grid_info['run_name']}/ioc/")
            output_dir_name = os.path.join(dirname, f"{grid_info['project_name']}/outputs/{grid_info['run_name']}/ioc/")
            just_file_name = f"{grid_info['run_name']}-{batch_name}_idx{ioc_idx}.jsonl"
            history_file_name = f"history-{grid_info['run_name']}-{batch_name}-idx{ioc_idx}.json"
        else:
            input_dir_name = os.path.join(dirname,f"{grid_info['project_name']}/inputs/{grid_info['run_name']}/")
            output_dir_name = os.path.join(dirname,f"{grid_info['project_name']}/outputs/{grid_info['run_name']}/")
            just_file_name = f"{grid_info['run_name']}-{batch_name}.jsonl"
            history_file_name = f"history-{grid_info['run_name']}-{batch_name}.json"

        # making said files, such and such ==================================================
        inputs_path = os.path.join(input_dir_name, just_file_name)
        os.makedirs(os.path.dirname(inputs_path), exist_ok=True)

        history_path = os.path.join(output_dir_name, history_file_name)
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        if os.path.exists(history_path):
            with open(history_path, 'r') as file:
                history = json.load(file)
        else:
            history = {}

        # constructing the requests ==================================================
        requests = []
        this_batch_history = [[] for _ in range(len(input_prompts))]
        
        # if idx of conversation is not 0, meaning there's a conversation.
        # update `history` and `input_prompts` from said history
        if config["in_one_context"]:
            new_input_prompts = [row[ioc_idx] for row in input_prompts]
            # TODO: update this_batch_history from whole history
            # TODO: update input_prompts accordingly
            input_prompts = new_input_prompts

        for prompt_idx, prompt in enumerate(input_prompts):
            this_batch_history[prompt_idx].append({"user" : prompt})
            requests.append({
                "custom_id": batch_name + f"-id{str(prompt_idx)}",
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

        with open(inputs_path, "w+") as f:
            for item in requests:
                json.dump(item, f)
                f.write('\n')

        history[batch_name] = this_batch_history
        with open(history_path, 'w') as file:
            json.dump(history, file, indent=4)

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
        
        batch_obj_id = "YEEEEEEHAW" + batch_name

        # TODO revert back to this
        # print(f"batch created! batch id : {batch_obj.id}", end=" ---- ")
        print(f"batch created! batch id : {batch_obj_id}", end=" ---- ")

        if(config["in_one_context"]):
            batchid_path = f"{grid_info['project_name']}/outputs/{grid_info['run_name']}/ioc/ioc_batch_ids-idx{ioc_idx}.txt"
        else:
            batchid_path = f"{grid_info['project_name']}/outputs/{grid_info['run_name']}/batch_ids.txt"
        
        print(f"saving in: {batchid_path}")
        batchid_file = os.path.join(os.path.dirname(__file__), batchid_path)
        os.makedirs(os.path.dirname(batchid_file), exist_ok=True)

        with open(batchid_file, "a") as bidf:
            # TODO revert back to this
            # bidf.write(batch_obj.id + "\n")
            bidf.write(batch_obj_id + "\n")
    
    def is_list_of_batch_all_done(self, batch_numbers):
        return True
        # statuses = [self.model.batches.retrieve(bn).status for bn in batch_numbers]
        # return all(batch == "completed" for batch in statuses)

    def extract_batch_results(self, project_name, run_name, batch_numbers):
        # TODO: update to the new history scheme
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
                            "messages" : data["response"]["body"]["choices"][0]["message"]["content"],
                        })
                    else:
                        quickview[prefix] = [{
                            "id" : data["custom_id"].split("_")[-1],
                            "messages" : data["response"]["body"]["choices"][0]["message"]["content"],
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
