import argparse
import commons

parser = argparse.ArgumentParser()
parser.add_argument("result_path", type=str)
parser.add_argument('-m', "--method_name", required=True, type=str)
args = parser.parse_args()

commons.submit_poses(method=args.method_name, path=args.result_path)
