import os
from model_manager import ModelManager

package_dir = os.path.dirname(__file__)
with open(os.path.join(package_dir, 'us_gender_percentage_pay.json'), "r") as file:
    print("file loaded, tinggal benerin aja")
    # TODO: reformat the loaded file to look like the grid example

grid = {
    "grid_info" : {
        "project_name" : __package__,
        "run_name" : "test_run222",
    },
    "data" : [
        {
            "config" : {
                "model_name" : "gpt-4o-mini",
                "in_one_context" : False,
                "args" : {
                    "temperature" : 1,
                    "top_p" : 1,
                }
            },
            "input_prompts" : [
                "What is the capital of China",
            ]
        },
        {
            "config" : {
                "model_name" : "gpt-4o-mini",
                "in_one_context" : False,
                "args" : {
                    "temperature" : 0,
                    "top_p" : 1,
                }
            },
            "input_prompts" : [
                "What is the capital of Lebanon",
            ]
        },
        {
            "config" : {
                "model_name" : "gpt-4o-mini",
                "in_one_context" : False,
                "args" : {
                    "temperature" : 1,
                    "top_p" : 0,
                }
            },
            "input_prompts" : [
                "What is the capital of Slovakia",
            ]
        },
    ]
}

modman = ModelManager()
modman.run_grid(grid)