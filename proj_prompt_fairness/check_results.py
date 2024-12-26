from model_manager import CustomOpenAIAgent
import argparse

parser = argparse.ArgumentParser(description="run name?")
parser.add_argument("--run_name", type=str, required=True, help="run name. make sure run batch is submitted though")

args = parser.parse_args()
run_name = args.run_name

print("run name gannn", run_name)

# model = CustomOpenAIAgent()
# model.check_and_download_batches(batch_ids, package_name = __package__)
