"""
Build a compact JSON for the Chinese website by merging occupations with AI exposure scores.

Reads cn/occupations.json (for stats) and cn/scores.json (for AI exposure).
Writes site-cn/data.json.

Usage:
    uv run python cn/build_site_data.py
"""

import json
import os


def main():
    # Load AI exposure scores
    with open("cn/scores.json") as f:
        scores_list = json.load(f)
    scores = {s["slug"]: s for s in scores_list}

    # Load occupations
    with open("cn/occupations.json") as f:
        occupations = json.load(f)

    # Merge
    data = []
    for occ in occupations:
        slug = occ["slug"]
        score = scores.get(slug, {})
        data.append({
            "title": occ["title"],
            "slug": slug,
            "category": occ["category"],
            "pay": occ.get("median_pay_annual"),
            "jobs": occ.get("num_jobs"),
            "outlook": occ.get("outlook_pct"),
            "outlook_desc": occ.get("outlook_desc", ""),
            "education": occ.get("education", ""),
            "exposure": score.get("exposure"),
            "exposure_rationale": score.get("rationale"),
            "url": "",
        })

    os.makedirs("site-cn", exist_ok=True)
    with open("site-cn/data.json", "w") as f:
        json.dump(data, f, ensure_ascii=False)

    print(f"Wrote {len(data)} occupations to site-cn/data.json")
    total_jobs = sum(d["jobs"] for d in data if d["jobs"])
    print(f"Total jobs represented: {total_jobs:,}")


if __name__ == "__main__":
    main()
