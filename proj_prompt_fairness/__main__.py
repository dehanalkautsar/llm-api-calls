import os
from model_manager import ModelManager

package_dir = os.path.dirname(__file__)
with open(os.path.join(package_dir, 'us_gender_percentage_pay.json'), "r") as file:
    print("file loaded, tinggal benerin aja")

examp = [
    {
        "config" : {
            "project_name" : __package__,
            "model_name" : "gpt-4o-mini",
            "in_one_context" : False,
            "args" : {
                "temperature" : 1,
                "top_p" : 1,
            }
        },
        "input_prompts" : [
            "What is the capital of Indonesia",
        ]
    },
        {
        "config" : {
            "project_name" : __package__,
            "model_name" : "gpt-4o-mini",
            "in_one_context" : False,
            "args" : {
                "temperature" : 0,
                "top_p" : 1,
            }
        },
        "input_prompts" : [
            "What is the capital of Spain",
        ]
    },
        {
        "config" : {
            "project_name" : __package__,
            "model_name" : "gpt-4o-mini",
            "in_one_context" : False,
            "args" : {
                "temperature" : 1,
                "top_p" : 0,
            }
        },
        "input_prompts" : [
            "What is the capital of Japan",
        ]
    },
]

modman = ModelManager()
modman.run_grid(examp)