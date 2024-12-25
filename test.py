import json
from openai import OpenAI
client = OpenAI()

batch = client.batches.retrieve("batch_676b91e1eddc8190bf405f41a73026fd")

# print(batch)
ofile = client.files.retrieve("file-1Yg8VeHCH89avsengRMbY3")
file_content = client.files.content("file-1Yg8VeHCH89avsengRMbY3")

def get_prefix_from_id(id_string):
    spl = id_string.split("_")
    return "_".join(spl[:-1])

quickview = {}
with open("quick_results.jsonl", "wb") as f:
    # dct = json.loads()
    dct_strings = file_content.content.decode('utf-8').strip().split("\n")
    for dct_string in dct_strings:
        data = json.loads(dct_string)
        prefix = get_prefix_from_id(data["custom_id"])
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

print(json.dumps(quickview, indent=4))