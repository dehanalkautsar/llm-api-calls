import os, argparse
from model_manager import CustomOpenAIAgent

parser = argparse.ArgumentParser(description="run name?")
parser.add_argument("--run_name", type=str, required=True, help="run name. make sure run batch is submitted though")

args = parser.parse_args()
run_name = args.run_name

batch_ids_file = os.path.join(os.path.dirname(__file__), run_name, "outputs/sep_batchids/sep_batch_ids.txt")
model = CustomOpenAIAgent()
model.extract_batch_results(__package__, run_name, batch_ids_file)
