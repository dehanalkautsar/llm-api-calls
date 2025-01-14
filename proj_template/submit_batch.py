import os, json
from model_manager import ModelManager

grid = {
    "grid_info" : {
        "project_name" : __package__,
        "run_name" : "test_run2",
    },
    "batches" : [
        {
            "config" : {
                "tag" : "runone",
                "model_name" : "gpt-4o-mini",
                "in_one_context" : True,
                "args" : {
                    "temperature" : 0,
                    "top_p" : 1,
                }
            },
            "input_prompts" : [
                ["What is the capital of Malaysia", "what is the currency", "what is the language"],
                ["What is the capital of Indonesia", "what is the currency", "what is the language"],
                ["What is the capital of Singapore", "what is the currency", "what is the language"],
            ]
        },
        {
            "config" : {
                "tag" : "runtwo",
                "model_name" : "gpt-4o-mini",
                "in_one_context" : False,
                "args" : {
                    "temperature" : 1,
                    "top_p" : 1,
                }
            },
            "input_prompts" : [
                "What is the capital of Slovakia",
                "What is the capital of Ukraine",
                "What is the capital of Russia",
            ]
        },
        {
            "config" : {
                "tag" : "runthree",
                "model_name" : "gpt-4o-mini",
                "in_one_context" : True,
                "args" : {
                    "temperature" : 0,
                    "top_p" : 0,
                }
            },
            "input_prompts" : [
                ["What is the capital of Vietnam", "what is the currency"],
                ["What is the capital of Thailand", "what is the currency"],
                ["What is the capital of Laos", "what is the currency"],
            ]
        },
        {
            "config" : {
                "tag" : "runfour",
                "model_name" : "gpt-4o-mini",
                "in_one_context" : False,
                "args" : {
                    "temperature" : 1,
                    "top_p" : 1,
                }
            },
            "input_prompts" : [
                "What is the capital of Italy",
                "What is the capital of Ireland",
                "What is the capital of Spain",
            ]
        },
    ]
}

grid_path = f"{__package__}/{grid['grid_info']['run_name']}/submitted_grid.json"
os.makedirs(os.path.dirname(grid_path), exist_ok=True)
with open(grid_path, "w") as gg:
    json.dump(grid, gg, indent=4)

modman = ModelManager()
modman.run_grid(grid)