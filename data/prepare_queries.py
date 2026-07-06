"""
Turn CelebA's 40 binary attributes into natural-language queries, and turn its
5-point facial landmarks into region bounding boxes so we can later measure
"did FocusVAE reconstruct the region this query cares about well?"

Only attributes that map cleanly onto a facial region are used (e.g.
"Smiling" -> mouth). Attributes with no clear spatial grounding (e.g.
"Attractive", "Young") are skipped — they wouldn't give the relevance map
anything meaningful to point at.

Usage:
    python data/prepare_queries.py --root data/celeba
Produces:
    data/celeba/queries.csv        (image_id, query, attribute, label, region)
    data/celeba/region_boxes.csv   (image_id, region, x1, y1, x2, y2)
"""
import argparse
from pathlib import Path

import pandas as pd

# attribute_name -> (natural language query template, region name)
ATTR_TO_QUERY = {
    "Smiling":            ("is the person smiling?", "mouth"),
    "Mouth_Slightly_Open": ("is the person's mouth open?", "mouth"),
    "Wearing_Lipstick":    ("is the person wearing lipstick?", "mouth"),
    "Mustache":            ("does the person have a mustache?", "mouth"),
    "No_Beard":            ("does the person have a beard?", "mouth"),
    "Eyeglasses":          ("is the person wearing glasses?", "eyes"),
    "Narrow_Eyes":         ("does the person have narrow eyes?", "eyes"),
    "Bags_Under_Eyes":     ("does the person have bags under their eyes?", "eyes"),
    "Arched_Eyebrows":     ("does the person have arched eyebrows?", "eyes"),
    "Bushy_Eyebrows":      ("does the person have bushy eyebrows?", "eyes"),
    "Big_Nose":            ("does the person have a big nose?", "nose"),
    "Pointy_Nose":         ("does the person have a pointy nose?", "nose"),
    "Bangs":               ("does the person have bangs?", "forehead"),
    "Wearing_Hat":         ("is the person wearing a hat?", "forehead"),
}

# region -> which landmark columns define its bounding box, and a margin
# (in pixels, on the standard 178x218 aligned CelebA crop) added around them
REGION_LANDMARKS = {
    "mouth":    (["leftmouth_x", "leftmouth_y", "rightmouth_x", "rightmouth_y"], 14),
    "eyes":     (["lefteye_x", "lefteye_y", "righteye_x", "righteye_y"], 14),
    "nose":     (["nose_x", "nose_y", "nose_x", "nose_y"], 18),
    "forehead": (["lefteye_x", "lefteye_y", "righteye_x", "righteye_y"], 0),  # shifted up below
}


def region_box(row, region):
    cols, margin = REGION_LANDMARKS[region]
    xs = [row[cols[0]], row[cols[2]]]
    ys = [row[cols[1]], row[cols[3]]]
    x1, x2 = min(xs) - margin, max(xs) + margin
    y1, y2 = min(ys) - margin, max(ys) + margin
    if region == "forehead":
        # forehead sits above the eyes: shift the eye box upward
        h = y2 - y1
        y1, y2 = y1 - 1.6 * h, y1 - 0.2 * h
    return x1, y1, x2, y2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data/celeba")
    args = ap.parse_args()
    root = Path(args.root)

    attr = pd.read_csv(root / "list_attr_celeba.csv")
    lmk = pd.read_csv(root / "list_landmarks_align_celeba.csv")
    attr = attr.merge(lmk, on="image_id")

    kept_ids = {p.name for p in (root / "images").glob("*.jpg")}
    attr = attr[attr["image_id"].isin(kept_ids)].reset_index(drop=True)

    query_rows = []
    box_rows = {}  # (image_id, region) -> box, deduped
    for _, row in attr.iterrows():
        img_id = row["image_id"]
        for attribute, (template, region) in ATTR_TO_QUERY.items():
            label = 1 if row[attribute] == 1 else 0
            query_rows.append({
                "image_id": img_id,
                "query": template,
                "attribute": attribute,
                "label": label,
                "region": region,
            })
            key = (img_id, region)
            if key not in box_rows:
                x1, y1, x2, y2 = region_box(row, region)
                box_rows[key] = {"image_id": img_id, "region": region,
                                  "x1": x1, "y1": y1, "x2": x2, "y2": y2}

    pd.DataFrame(query_rows).to_csv(root / "queries.csv", index=False)
    pd.DataFrame(list(box_rows.values())).to_csv(root / "region_boxes.csv", index=False)
    print(f"Wrote {len(query_rows)} queries and {len(box_rows)} region boxes "
          f"for {len(kept_ids)} images to {root}")


if __name__ == "__main__":
    main()
