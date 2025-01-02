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
        for job in grid["batches"]:
            if job["config"]["in_one_context"]:
                self._is_ioc_input_shape_valid(job["input_prompts"])
        
        # submit all first
        for jobidx, job in enumerate(grid["batches"]):
            if job["config"]["in_one_context"]:
                in_one_context_idxes.append(jobidx)
            agent = self._instantiate_model_based_on_name(job["config"]["model_name"])
            agent.run_batch(grid["grid_info"],
                            job["config"],
                            job["input_prompts"])
        print("grid finished submitting/running. continuing with in one context prompts...")
        if len(in_one_context_idxes) > 0:
            print("in one context prompts detected, running...")
            longest_ioc = max([len(grid["batches"][i]["input_prompts"][0]) for i in in_one_context_idxes])

            for i in range(longest_ioc):
                prev_batchid_path = f"{grid['grid_info']['project_name']}/{grid['grid_info']['run_name']}/outputs/ioc/ioc_batch_ids-idx{i}.txt"
                prev_batchid_file = os.path.join(os.path.dirname(__file__), prev_batchid_path)
                
                with open(prev_batchid_file, "r") as bf:
                    batchids = [line.strip().split(":")[-1] for line in bf]
                
                tries = 0
                while(not agent.is_list_of_batch_all_done(batchids) and tries < MAX_CHECK_TRIES):
                    print("waiting for all the ioc batches to finish...")
                    tries += 1
                    print("sleeping for 2s (change to 3 minutes ntar)")
                    sleep(2)
                
                if tries == 5:
                    raise TimeoutError("waited 5 times each 3 minutes already")

                # batch is done, extract and run next one                
                # extract
                print(f"=========================================")
                print(f"message index {i} is done! extracting...")
                agent.extract_batch_results(
                    grid['grid_info']['project_name'],
                    grid['grid_info']['run_name'],
                    prev_batchid_file
                )
                print(f"message index {i} extracted")
                
                # run next one
                for j in in_one_context_idxes:
                    if i+1 < len(grid["batches"][j]["input_prompts"][0]):
                        print(f"starting next one (message index {i+1}) for batch index {j} in {grid['grid_info']['run_name']}")
                        agent.run_batch(grid["grid_info"],
                                        grid["batches"][j]["config"],
                                        grid["batches"][j]["input_prompts"],
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
            input_dir_name = os.path.join(dirname, f"{grid_info['project_name']}/{grid_info['run_name']}/inputs/ioc/")
            output_dir_name = os.path.join(dirname, f"{grid_info['project_name']}/{grid_info['run_name']}/outputs/ioc/")
            just_file_name = f"{grid_info['run_name']}-{batch_name}_idx{ioc_idx}.jsonl"
            history_file_name = f"history-{grid_info['run_name']}-ioc.json"
        else:
            input_dir_name = os.path.join(dirname,f"{grid_info['project_name']}/{grid_info['run_name']}/inputs/")
            output_dir_name = os.path.join(dirname,f"{grid_info['project_name']}/{grid_info['run_name']}/outputs/")
            just_file_name = f"{grid_info['run_name']}-{batch_name}.jsonl"
            history_file_name = f"history-{grid_info['run_name']}-sep.json"

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
        messages = []
        
        # if idx of conversation is not 0, meaning there's a conversation.
        # update `history` and `input_prompts` from said history            

        if config["in_one_context"]:
            input_prompts = [row[ioc_idx] for row in input_prompts]        
        
        for pr_idx, pr in enumerate(input_prompts):
                if (ioc_idx == 0):
                    messages.append([{
                        "role" : "user",
                        "content" : pr,
                    }])
                else:
                    messages.append(
                        history[batch_name][pr_idx] +
                        [{
                            "role" : "user",
                            "content" : pr,
                        }]
                    )

        for msg_idx, msg in enumerate(messages):
            requests.append({
                "custom_id": batch_name + f"-id{str(msg_idx)}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": config["model_name"],
                    "messages": msg,
                    **config.get("args", {})
                }
            })

        with open(inputs_path, "w+") as f:
            for item in requests:
                json.dump(item, f)
                f.write('\n')

        history[batch_name] = messages
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
        
        # TODO revert back to this
        # bid = batch_obj.id
        bid = "batch_676b91e1eddc8190bf405f41a73026fd"
        batch_obj_id = f"{batch_name}:{bid}"

        print(f"batch created! batch id : {bid}", end=" ---- ")

        if(config["in_one_context"]):
            batchid_path = f"{grid_info['project_name']}/{grid_info['run_name']}/outputs/ioc/ioc_batch_ids-idx{ioc_idx}.txt"
        else:
            batchid_path = f"{grid_info['project_name']}/{grid_info['run_name']}/outputs/batch_ids.txt"
        
        print(f"saving in: {batchid_path}")
        batchid_file = os.path.join(os.path.dirname(__file__), batchid_path)
        os.makedirs(os.path.dirname(batchid_file), exist_ok=True)

        with open(batchid_file, "a") as bidf:
            bidf.write(batch_obj_id + "\n")
    
    def is_list_of_batch_all_done(self, batch_numbers):
        statuses = [self.model.batches.retrieve(bn).status for bn in batch_numbers]
        return all(batch == "completed" for batch in statuses)

    def extract_batch_results(self, project_name, run_name, batch_ids_file):
        if os.path.exists(batch_ids_file):
            dirname = os.path.dirname(__file__)
            # history file path
            if '/ioc/' in str(batch_ids_file):
                output_dir_name = os.path.join(dirname, f"{project_name}/{run_name}/outputs/ioc/")
                history_file_name = f"history-{run_name}-ioc.json"
            else:
                output_dir_name = os.path.join(dirname, f"{project_name}/{run_name}/outputs/")
                history_file_name = f"history-{run_name}-sep.json"
            
            history_path = os.path.join(output_dir_name, history_file_name)
            if os.path.exists(history_path):
                # load history
                with open(history_path, 'r') as h_file:
                    history = json.load(h_file)
                with open(batch_ids_file, "r") as b_file:
                    batch_name_and_ids = b_file.read().splitlines()

                for bnid in batch_name_and_ids:
                    bname_and_id = bnid.split(":")
                    batchname = bname_and_id[0]
                    batchid = bname_and_id[1]
                    batch = self.model.batches.retrieve(batchid)
                    if batch.status == "completed":
                        file_content = self.model.files.content(batch.output_file_id)
                        file_content_string = file_content.content.decode('utf-8')
                        dct_strings = file_content_string.strip().split("\n")
                        for idx, dct_string in enumerate(dct_strings):
                            data = json.loads(dct_string)
                            assistant_answer = data["response"]["body"]["choices"][0]["message"]["content"]
                            history[batchname][idx].append({
                                "role" : "assistant",
                                "content" : assistant_answer
                            })
                
                # save new history
                with open(history_path, 'w') as hfile:
                    json.dump(history, hfile, indent=4)
            else:
                raise FileNotFoundError(f"history file not found for {run_name}. make sure that experiment has been run")
        else:
            raise FileNotFoundError(f"batch ids file not found for {run_name}. make sure that experiment has been run")

    @staticmethod
    def get_prefix_from_id(id_string):
        spl = id_string.split("_")
        return "_".join(spl[:-1])
