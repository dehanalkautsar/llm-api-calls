import os
from model_manager import ModelManager

package_dir = os.path.dirname(__file__)
with open(os.path.join(package_dir, 'us_gender_percentage_pay.json'), "r") as file:
    print("file loaded, tinggal benerin aja")
    # TODO: reformat the loaded file to look like the grid example


# CONVENTION: PUT IN-ONE-CONTEXT-PROMPTS AT THE END!

grid = {
    "grid_info" : {
        "project_name" : __package__,
        "run_name" : "testgannnn",
    },
    "data" : [
        {
            "config" : {
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
                "What is the capital of Ukraine",
            ]
        },
                {
            "config" : {
                "model_name" : "gpt-4o-mini",
                "in_one_context" : True,
                "args" : {
                    "temperature" : 0,
                    "top_p" : 0,
                }
            },
            "input_prompts" : [
                ["What is the capital of Vietnam", "what is the currency", "what is the language"],
                ["What is the capital of Thailand", "what is the currency", "what is the language"],
            ]
        },
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
                "What is the capital of Italy",
                "What is the capital of Ireland",
            ]
        },
    ]
}

modman = ModelManager()
modman.run_grid(grid)