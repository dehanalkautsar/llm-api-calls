import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from time import sleep
from agents import CustomOpenAIAgent
load_dotenv()

OPENAI_MODELS = ["gpt-4o-mini"]
MAX_CHECK_TRIES = 1440
WAIT_TIME = 60

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
        
        print("grid finished submitting/running. continuing with in one context prompts (if any)")
        self.continue_openai_grid_ioc(grid, begin_index=1)

    def continue_openai_grid_ioc(self, grid, begin_index):
        agent = CustomOpenAIAgent()
        in_one_context_idxes = []
        for jobidx, job in enumerate(grid["batches"]):
            if job["config"]["in_one_context"]:
                in_one_context_idxes.append(jobidx)
        
        if len(in_one_context_idxes) > 0:
            print("in one context prompts detected, running...")
            longest_ioc = max([len(grid["batches"][i]["input_prompts"][0]) for i in in_one_context_idxes])

            for i in range(begin_index, longest_ioc+1):
                prev_batchid_path = f"{grid['grid_info']['project_name']}/{grid['grid_info']['run_name']}/outputs/ioc_batchids/ioc_batch_ids-idx{i-1}.txt"
                prev_batchid_file = os.path.join(os.path.dirname(__file__), prev_batchid_path)
                
                with open(prev_batchid_file, "r") as bf:
                    batchids = [line.strip().split(":")[-1] for line in bf]

                tries = 0
                while(not agent.is_list_of_batch_all_done(batchids) and tries < MAX_CHECK_TRIES):
                    print("waiting for all the ioc batches to finish...")
                    tries += 1
                    print(f"checking if finished every {WAIT_TIME} seconds, waiting for the batch to finish")
                    sleep(WAIT_TIME)
                
                if tries == MAX_CHECK_TRIES:
                    raise TimeoutError(f"waited {MAX_CHECK_TRIES} times each {WAIT_TIME} seconds already")

                # batch is done, extract and run next one                
                # extract
                print(f"=========================================")
                print(f"message index {i-1} is done! extracting...")
                agent.extract_batch_results(
                    grid['grid_info']['project_name'],
                    grid['grid_info']['run_name'],
                    prev_batchid_file
                )
                print(f"message index {i-1} extracted")
                
                # run next one
                for j in in_one_context_idxes:
                    if i < len(grid["batches"][j]["input_prompts"][0]):
                        print(f"starting next one (message index {i}) for batch index {j} in {grid['grid_info']['run_name']}")
                        agent.run_batch(grid["grid_info"],
                                        grid["batches"][j]["config"],
                                        grid["batches"][j]["input_prompts"],
                                        i)

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
