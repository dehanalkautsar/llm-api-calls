from model_manager import CustomOpenAIAgent
# import argparse
# 
# parser = argparse.ArgumentParser(description="provide batch number")
# parser.add_argument("--batch_number", type=int, required=True, help="the batch number to process.")
# 
# args = parser.parse_args()
# batch_number = args.batch_number
# 
batch_ids = [
    "batch_676b91e1eddc8190bf405f41a73026fd"
]

model = CustomOpenAIAgent()
model.check_and_download_batches(batch_ids, package_name = __package__)
