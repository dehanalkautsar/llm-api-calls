import os, argparse
from model_manager import CustomOpenAIAgent

parser = argparse.ArgumentParser(description="run name?")
parser.add_argument("--run_name", type=str, required=True, help="run name. make sure run batch is submitted though")

args = parser.parse_args()
run_name = args.run_name

batch_ids_file = os.path.join(os.path.dirname(__file__), "outputs", run_name, "batch_ids.txt")
if os.path.exists(batch_ids_file):
    with open(batch_ids_file, "r") as file:
        batch_ids = file.read().splitlines()
        # TODO: other LLM providers i.e. non OpenAI
        model = CustomOpenAIAgent()
        model.check_and_download_batches(__package__, run_name, batch_ids)
else:
    raise FileNotFoundError(f"batch_ids.txt file not found for {run_name}. make sure that experiment has been run")


