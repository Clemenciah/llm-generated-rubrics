import subprocess
import os
import glob


DATASET = "SummEval"

definition_glob = os.path.join("results", "definition", DATASET, "*.json")
files = [os.path.basename(p) for p in glob.glob(definition_glob)]

for setting_name in files:
    for use_sim in ("True", "False"):
        cmd = [
            "poetry", "run", "python",
            os.path.join(".", "scripts", "tag_scores.py"),
            "--data", DATASET,
            "--setting_name", setting_name,
            "--use_similarity", use_sim,
            "--similarity_threshold", "0.82",
            "--meaningfulness_threshold", "1",
            "--device", "cpu"
        ]
        print(" ".join(cmd))
        print("=" * 100)
        subprocess.run(cmd, check=True)