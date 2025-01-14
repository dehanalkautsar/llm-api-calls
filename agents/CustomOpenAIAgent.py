import os, json
from typing import List, Union
from openai import OpenAI

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
            batch_name = f"{config["tag"]}-{config["model_name"].replace("-", "_")}-{in1ctx}-{arg_string}"
        else:
            batch_name = f"{config["tag"]}-{config["model_name"].replace("-", "_")}-{in1ctx}"

        dirname = os.getcwd()
        project_run_path = os.path.join(dirname, f"{grid_info['project_name']}/{grid_info['run_name']}")
        input_dir_name = os.path.join(project_run_path, "inputs")
        output_dir_name = os.path.join(project_run_path, "outputs")
        
        batchids_dir = ("ioc" if config["in_one_context"] else "sep") + "_batchids"
        if (config["in_one_context"]):
            jsonl_dir = "ioc_jsonls/" + f"{grid_info['run_name']}-{batch_name}"
        else:
            jsonl_dir = "sep_jsonls"
        
        jsonl_file_name = f"{grid_info['run_name']}-{batch_name}{"_idx"+str(ioc_idx) if config["in_one_context"] else ""}.jsonl"
        batchid_file_name = f"ioc_batch_ids-idx{ioc_idx}.txt" if config["in_one_context"] else "sep_batch_ids.txt"
        history_file_name = f"history-{grid_info['run_name']}-{"ioc" if config["in_one_context"] else "sep"}.json"

        # making said files, such and such ==================================================
        jsonl_file = os.path.join(input_dir_name, jsonl_dir, jsonl_file_name)
        os.makedirs(os.path.dirname(jsonl_file), exist_ok=True)

        batchid_file = os.path.join(output_dir_name, batchids_dir, batchid_file_name)
        os.makedirs(os.path.dirname(batchid_file), exist_ok=True)

        history_file = os.path.join(output_dir_name, "history", history_file_name)
        os.makedirs(os.path.dirname(history_file), exist_ok=True)

        if os.path.exists(history_file):
            with open(history_file, 'r') as file:
                history = json.load(file)
        else:
            history = {}
        
        # constructing the requests ==================================================
        requests = []
        messages = []

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

        with open(jsonl_file, "w+") as f:
            for item in requests:
                json.dump(item, f)
                f.write('\n')

        history[batch_name] = messages
        with open(history_file, 'w') as file:
            json.dump(history, file, indent=4)

        # replace if there already exists
        for uploaded_files in self.model.files.list():
            if uploaded_files.filename == just_file_name:
                print(f"{just_file_name} already exists, replacing...")
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
                "project_name" : grid_info['project_name'],
                "run_name" : grid_info['run_name']
            }
        )
        
        bid = batch_obj.id
        batch_obj_id = f"{batch_name}:{bid}"

        print(f"batch created! batch id : {bid}", end=" ---- ")
        print(f"saving in: {batchid_file}")
        with open(batchid_file, "a") as bidf:
            bidf.write(batch_obj_id + "\n")
    
    def is_list_of_batch_all_done(self, batch_numbers):
        statuses = [self.model.batches.retrieve(bn).status for bn in batch_numbers]
        return all(batch == "completed" for batch in statuses)

    def extract_batch_results(self, project_name, run_name, batch_ids_file):
        if os.path.exists(batch_ids_file):
            dirname = os.path.join(os.getcwd(), project_name) 
            history_file_name = f"history-{run_name}-{"ioc" if "ioc" in str(batch_ids_file) else "sep"}.json"
            history_path = os.path.join(dirname,
                                        f"{run_name}/outputs/history",
                                        history_file_name)

            if os.path.exists(history_path):
                # load history
                with open(history_path, 'r') as h_file:
                    history = json.load(h_file)
                with open(batch_ids_file, "r") as b_file:
                    batch_name_and_ids = b_file.read().splitlines()

                write = True
                for bnid in batch_name_and_ids:
                    bname_and_id = bnid.split(":")
                    batchname = bname_and_id[0]
                    batchid = bname_and_id[1]
                    batch = self.model.batches.retrieve(batchid)
                    if batch.status == "completed":
                        print(f"{batchname} - {batchid} is done")
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
                    else:
                        print(f"{batchname} - {batchid} is not done. status: {batch.status}")
                        write = False
                
                # save new history
                if(write):
                    print("all batch done. writing..")
                    with open(history_path, 'w') as hfile:
                        json.dump(history, hfile, indent=4)
                else:
                    print("not all is done. not writing results")
            else:
                raise FileNotFoundError(f"history file not found for {run_name}. make sure that experiment has been run")
        else:
            raise FileNotFoundError(f"batch ids file not found for {run_name}. make sure that experiment has been run")

    @staticmethod
    def get_prefix_from_id(id_string):
        spl = id_string.split("_")
        return "_".join(spl[:-1])
