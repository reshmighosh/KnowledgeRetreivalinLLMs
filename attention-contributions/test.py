import os
import json
dirs = os.listdir("/work/pi_dhruveshpate_umass_edu/rseetharaman_umass_edu/repo-for-paper/attention-contributions-llama/outputs/llama-2-base")
failed = set()
for d in dirs:
    if 'counterfact' not in d:
        continue
    path = os.path.join("/work/pi_dhruveshpate_umass_edu/rseetharaman_umass_edu/repo-for-paper/attention-contributions-llama/outputs/llama-2-base", d)
    # print(f"Checking {path}")
    if(len(os.listdir(f"{path}/plots")) != 977):
        failed.add(path)
    objs = json.load(open(f"{path}/outputs/full_data.json"))
    if(len(objs) != 977):
        failed.add(path)
    if(len(os.listdir(f"{path}/data")) != 977):
        failed.add(path)

for d in dirs:
    if 'counterfact' in d:
        continue
    path = os.path.join("/work/pi_dhruveshpate_umass_edu/rseetharaman_umass_edu/repo-for-paper/attention-contributions-llama/outputs/llama-2-base", d)
    # print(f"Checking {path}")
    if(len(os.listdir(f"{path}/plots")) != 978):
        failed.add(path)
    objs = json.load(open(f"{path}/outputs/full_data.json"))
    if(len(objs) != 978):
        failed.add(path)
    if(len(os.listdir(f"{path}/data")) != 978):
        failed.add(path)

if len(failed) == 0:
    print(f"All tests passed")

for f in failed:
    print(f)