from dotenv import load_dotenv
import argparse
import os
import sys
import json
from datetime import datetime, UTC


# --- Must set device BEFORE loading embedding libs (done via env var) ---
def set_device_env(device: str):
    # Force CPU by hiding GPUs from torch if requested
    if device.lower() == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # else leave as-is; SentenceTransformer will use GPU if available


############### Directories #########################
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

datasets_dir = os.path.join(parent_dir, 'datasets')
data_root_dir = os.path.join(parent_dir, 'data_information')

############### Load API keys  ###############
load_dotenv()

############### Project imports #####################
from data.data_loader import Dataset
from util.similarity_tools import merge_similar_metrics  # uses sentence-transformers or string fallback
import pandas as pd


#################### Helpers ########################

def load_definition_rows(dataset: str, setting_name: str):
    """
    Returns a flat list of dicts with keys at least: name, description.
    """
    path = os.path.join("results", "definition", dataset, setting_name)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        rows = [m for chunk in raw for m in chunk]
    else:
        rows = raw

    # ensure minimal fields
    cleaned = []
    for m in rows:
        cleaned.append({
            "name": m.get("name", "").strip(),
            "description": m.get("description", "").strip(),
            "scale": m.get("scale", ""),
            "instruction": m.get("instruction", "")
        })
    return cleaned


def load_tags(dataset: str, setting_name: str):
    """
    Loads list like: [{"name": "...", "tags": ["...", ...]}, ...
    """
    path = os.path.join("results", "tag", dataset, setting_name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing tag file for this run:\n{path}\nRun the tagging step first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize(s: str) -> str:
    import re
    return re.sub(r'[^a-z0-9]+', '', s.lower())


def build_alias_map(teacher_names):
    """
    Build a forgiving mapper:
    - exact canonical names
    - split composites ("/", "vs", "&") to allow partial matches like 'Helpfulness' -> 'Helpfulness/Understanding'
    - add a few generic synonyms if the target exists in the teacher list
    """
    import re
    canon = {normalize(n): n for n in teacher_names}
    alias = {}
    for name in teacher_names:
        parts = re.split(r'\s*(?:/|vs\.?|&)\s*', name, flags=re.I)
        for p in parts:
            k = normalize(p)
            if k and k != normalize(name):
                alias[k] = name

    # generic synonyms that map IF the target is in the teacher set
    def maybe_add(src, target):
        k = normalize(src)
        if target in teacher_names:
            alias[k] = target

    # Common
    maybe_add("accuracy", next((t for t in teacher_names if "Correctness" in t or "Completeness" in t), None) or "")
    maybe_add("clarity", next((t for t in teacher_names if "Coherence" in t or "Clarity" in t), None) or "")
    maybe_add("coherence", next((t for t in teacher_names if "Coherence" in t or "Clarity" in t), None) or "")
    maybe_add("readability", next((t for t in teacher_names if "Readability" in t), None) or "")
    maybe_add("helpfulness", next((t for t in teacher_names if "Helpfulness" in t), None) or "")
    maybe_add("succinct", next((t for t in teacher_names if "Succinct" in t or "Verbose" in t), None) or "")
    maybe_add("verbose", next((t for t in teacher_names if "Succinct" in t or "Verbose" in t), None) or "")
    maybe_add("informativeness",
              next((t for t in teacher_names if "Informativeness" in t or "Focus" in t or "Overlap" in t), None) or "")
    maybe_add("coverage",
              next((t for t in teacher_names if "Informativeness" in t or "Coverage" in t or "Overlap" in t),
                   None) or "")
    maybe_add("overlap", next((t for t in teacher_names if "Informativeness" in t or "Overlap" in t), None) or "")
    maybe_add("focus", next((t for t in teacher_names if "Informativeness" in t or "Focus" in t), None) or "")
    maybe_add("safety", next((t for t in teacher_names if "Safe" in t), None) or "")

    return canon, alias


def map_to_teacher(tag, canon, alias):
    k = normalize(tag)
    if k in canon:
        return canon[k]
    if k in alias and alias[k]:
        return alias[k]
    return None


def extract_components(filename: str):
    """
    Parse filenames like:
    model_name=gpt-4o-mini-use_system=True-use_sample=False-... .json
    into a dict of components.
    """
    if filename.endswith('.json'):
        filename = filename[:-5]
    parts = filename.split('=')
    result = {}
    for i in range(1, len(parts)):
        prev = parts[i - 1]
        curr = parts[i]
        key = prev.split('-')[-1]
        if i < len(parts) - 1:
            idx = curr.rfind('-')
            value = curr[:idx]
            parts[i] = curr[idx + 1:]
        else:
            value = curr
        result[key] = value
    return result


def compute_unseen_and_aux(tag_items, teacher_names):
    """
    Returns dict with:
      unseen_count, teacher_hit_count, teacher_total, coverage, diversity,
      new_aspects (sorted list)
    """
    canon, alias = build_alias_map(teacher_names)

    union_tags = set()
    for it in tag_items:
        union_tags.update(it.get("tags", []))

    # coverage / diversity work on the union set
    mapped_union = {map_to_teacher(t, canon, alias) for t in union_tags}
    teacher_hits = {m for m in mapped_union if m is not None}
    teacher_hit_count = len(teacher_hits)
    teacher_total = max(1, len(teacher_names))
    coverage = teacher_hit_count / teacher_total

    # diversity = 1 - precision (fraction of tags not in teacher)
    unique_tag_norm = {normalize(t) for t in union_tags}
    in_vocab = sum(1 for t in unique_tag_norm if map_to_teacher(t, canon, alias) is not None)
    diversity = 0.0 if not unique_tag_norm else 1 - (in_vocab / len(unique_tag_norm))

    # unseen per student metric: if NONE of its tags map to teacher -> unseen
    unseen_count = 0
    new_aspect_set = set()
    for it in tag_items:
        tags = it.get("tags", [])
        mapped = [map_to_teacher(t, canon, alias) for t in tags]
        if all(m is None for m in mapped):
            unseen_count += 1
        # collect new-aspect tags (those not mapping)
        for t, m in zip(tags, mapped):
            if m is None:
                new_aspect_set.add(t)

    return {
        "unseen_count": unseen_count,
        "teacher_hit_count": teacher_hit_count,
        "teacher_total": teacher_total,
        "coverage": coverage,
        "diversity": diversity,
        "new_aspects": sorted(new_aspect_set),
    }


######################## Main ########################

def main():
    parser = argparse.ArgumentParser(description="Compute tag-based summary scores for a run.")
    parser.add_argument('--data', choices=["USR", "SummEval", "SumPubMed", "MT-Bench", "CommonGen", "HelpSteer2"],
                        required=True, help='Dataset name')
    parser.add_argument('--setting_name', required=True,
                        help='Filename under results/definition/<dataset>/ to analyze (also used to load tags)')
    parser.add_argument('--use_similarity', type=lambda x: str(x).lower() == "true", default=True,
                        help='Use embedding-based description similarity for Unique (True/False)')
    parser.add_argument('--similarity_threshold', type=float, default=0.82,
                        help='Similarity threshold for merging duplicates (0..1)')
    parser.add_argument('--meaningfulness_threshold', type=int, default=1,
                        help='Keep clusters with size >= this (set to 1 for Unique counting)')
    parser.add_argument('--device', choices=["cpu", "gpu"], default="cpu",
                        help='Force CPU or allow GPU for embeddings')
    args = parser.parse_args()

    set_device_env(args.device)

    dataset = Dataset(datasets_dir, data_root_dir, args.data)
    teacher = dataset.ground_truth_metrics
    if args.data == "USR":
        teacher = teacher[:-1]

    if args.data == "SummEval":
        teacher = teacher[:-1]

    if args.data == "SumPubMed":
        teacher = teacher[:-1]
    teacher_names = [m.name for m in teacher]

    student_rows = load_definition_rows(args.data, args.setting_name)
    generated = len(student_rows)

    # Compute UNIQUE: merge duplicates via utility
    df = pd.DataFrame(student_rows, columns=["name", "description", "scale", "instruction"])
    merged_df = merge_similar_metrics(
        metric_description_df=df.copy(),
        use_similarity=args.use_similarity,
        similarity_threshold=args.similarity_threshold,
        meaningfulness_threshold=args.meaningfulness_threshold
    )
    unique_count = len(merged_df)

    # Load tags (for Unseen & coverage/diversity)
    tag_items = load_tags(args.data, args.setting_name)
    unseen_aux = compute_unseen_and_aux(tag_items, teacher_names)
    unseen_count = unseen_aux["unseen_count"]

    # Ratios
    unseen_over_generated = 0.0 if generated == 0 else unseen_count / generated
    unseen_over_unique = 0.0 if unique_count == 0 else unseen_count / unique_count

    # Filename components (for convenience columns)
    components = extract_components(args.setting_name)

    result = {
        "dataset": args.data,
        "setting_name": args.setting_name,
        **components,
        "generated": generated,
        "unique": unique_count,
        "unseen": unseen_count,
        "unseen_over_generated": unseen_over_generated,
        "unseen_over_unique": unseen_over_unique,
        "teacher_coverage": unseen_aux["coverage"],
        "diversity": unseen_aux["diversity"],
        "teacher_hit_count": unseen_aux["teacher_hit_count"],
        "teacher_total": unseen_aux["teacher_total"],
        "new_aspect_count": len(unseen_aux["new_aspects"]),
        "new_aspects": unseen_aux["new_aspects"],
        "use_similarity": args.use_similarity,
        "similarity_threshold": args.similarity_threshold,
        "meaningfulness_threshold": args.meaningfulness_threshold,
        "device": args.device,
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }


    out_dir = os.path.join("results", "tag", args.data, "scores", "embedding" if args.use_similarity else "name")
    os.makedirs(out_dir, exist_ok=True)


    json_out = os.path.join(out_dir, args.setting_name)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


    csv_out = os.path.join(out_dir, "_summary.csv")
    cols = [
        "dataset", "setting_name",
        "model_name", "use_system", "use_sample", "use_response", "number_of_shots", "use_temperature",
        "use_restriction",
        "generated", "unique", "unseen", "unseen_over_generated", "unseen_over_unique",
        "teacher_coverage", "diversity", "teacher_hit_count", "teacher_total",
        "new_aspect_count", "use_similarity", "similarity_threshold", "meaningfulness_threshold", "device", "timestamp"
    ]
    # Normalize missing columns in components
    for c in ["model_name", "use_system", "use_sample", "use_response", "number_of_shots", "use_temperature",
              "use_restriction"]:
        result.setdefault(c, None)

    df_row = pd.DataFrame([{k: result.get(k) for k in cols}])
    if os.path.exists(csv_out):
        df_all = pd.read_csv(csv_out)
        df_all = pd.concat([df_all, df_row], ignore_index=True)
        df_all.to_csv(csv_out, index=False)
    else:
        df_row.to_csv(csv_out, index=False)

    display_cols = ["dataset", "setting_name", "generated", "unique", "unseen", "unseen_over_generated",
                    "teacher_coverage", "diversity", "new_aspect_count"]
    print("\nTag score summary:")
    print(pd.DataFrame([{k: result[k] for k in display_cols}]))


if __name__ == '__main__':
    main()
