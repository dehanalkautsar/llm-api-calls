import os, argparse, json
from model_manager import ModelManager

parser = argparse.ArgumentParser(description="run name?")
parser.add_argument("--run_name", type=str, required=True, help="run name. make sure run batch is submitted though")

args = parser.parse_args()
run_name = args.run_name

# check which batches are ioc and needed to be checked
grid_path = f"{__package__}/{run_name}/submitted_grid.json"
with open(grid_path, "r") as gf:
    grid = json.load(gf)

conv_lengths = []
for batch in grid['batches']:
    if(type(batch['input_prompts'][0]) == list):
        conv_lengths.append(len(batch['input_prompts'][0]))

max_conv_length = max(conv_lengths)
ioc_output_dir = f"{__package__}/{run_name}/outputs/ioc"


modman = ModelManager()
for i in range(max_conv_length):
    batch_file = ioc_output_dir + f"/ioc_batch_ids-idx{i}.txt"
    if not(os.path.exists(batch_file)):
        completed = False
        print(f"file {batch_file} doesn't exist!, continuing from index {i}.")
        modman.continue_openai_grid_ioc(grid, i)
        break
    else:
        if(i == max_conv_length-1):
            print("ioc run is already completed")
    