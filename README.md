# llm-api-calls
ever wanted to evaluate an LLM by prompting it many times? (looking at you, AI researchers)
basically this lets you do it (including variations of the args like `temperature`, `top p`, whathaveyou)

## a work in progress
supported models:
- OpenAI

## hew 2 run in quick and ez steps (after installing the necessary modules)
### OpenAI
1. make an `.env` file, just copy `.env.example` and provide your own OpenAI API Key
2. make a project file (you can copy `proj_template`)
3. define your prompts in `submit_batch.py` just follow the format of the already existing example (and adjust the `run_name`)
4. `python -m proj_template.submit_batch` but change `proj_template` to your newly created project folder
5. wait
6. for non in-one-context batches, it wont check automatically. so you need to run `python -m proj_prompt_fairness.check_results --run_name test_run`
7. results are in `proj_name/outputs`
8. want to run again? just go back to step 3, and dont forget to change `run_name` so you dont overwrite the old run

### more models coming soon hehe

## in case you want to know how the projects and prompts and files are organized/named
### project
- a duplicate of `proj_template`. naming convention `proj_<something>`
- a project can have several runs

### run / grid
- a run is defined by a grid
- a grid : defines multiple batches

### batch
- a batch defines the specific prompts and arguments (in case you want to variate the temp, top p andwhatnot)
- supports single, separate prompts and in-one-context prompts (basically like in one chatroom) 

#### example separate contexts batch
```json
    {
        "config" : {
            "tag" : "batchone", // it says "tag" but this is really just the name of the batch lol
            "model_name" : "gpt-4o-mini", // the model name
            "in_one_context" : False, // meaning no chat history whatsoever is involved
            "args" : { // what you want to pass into the openai api call
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
```
for separate contexts, you'll have to check manually if your batch has finished running. (see how to run). the OpenAI people said that it should finish within 24h

#### example in-one-context batch
```json
{
    "config" : {
        "tag" : "batchtwo",
        "model_name" : "gpt-4o-mini",
        "in_one_context" : True, // prompt in one context window for each row (see input_prompts). basically includes a chat history
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
```
for in one context prompts, it will automatically check for you every 3 minutes (max tries 5 times, after that it'll raise an error lol). and if the first message is done, it'll automatically make requests of the next message, including the chat history

### result files
for separate prompts, make sure you've run the check and download (see how2run). after that, it's supposed to show in `proj_name/outputs`, specifically `history-run_name-sep.json`. for in-one-context, its in `proj_name/outputs/ioc`, name of file ends with `ioc` instead of `sep`

this file is automatically updated. you dont have to edit this. it also gives automated names to each run to preserve the run's "identity"
#### and this automated run is named how? im happy you asked:
`tag-modelname-('sep' or 'ioc')-arguments`

example results

```json
{
    "batchone-gpt_4o_mini-sep-temperature_1-top_p_1": [
        [
            {
                "role": "user",
                "content": "What is the capital of Slovakia"
            },
            {
                "role": "assistant",
                "content": "The capital of Slovakia is Bratislava."
            }
        ],
        [
            {
                "role": "user",
                "content": "What is the capital of Ukraine"
            },
            {
                "role": "assistant",
                "content": "The capital of Ukraine is Kyiv."
            }
        ],
        [
            {
                "role": "user",
                "content": "What is the capital of Russia"
            },
            {
                "role": "assistant",
                "content": "The capital of Russia is Moscow."
            }
        ]
    ],
    "batchtwo-gpt_4o_mini-sep-temperature_1-top_p_1": [
        [
            {
                "role": "user",
                "content": "What is the capital of Italy"
            },
            {
                "role": "assistant",
                "content": "The capital of Italy is Rome."
            }
        ],
        [
            {
                "role": "user",
                "content": "What is the capital of Ireland"
            },
            {
                "role": "assistant",
                "content": "The capital of Ireland is Dublin."
            }
        ],
        [
            {
                "role": "user",
                "content": "What is the capital of Spain"
            },
            {
                "role": "assistant",
                "content": "The capital of Spain is Madrid."
            }
        ]
    ]
}
```