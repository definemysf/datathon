#!/usr/bin/env python3
"""Race-focused DualGuard pipeline with a trained local matching model.

This fixes the earlier PoC's main weakness: the audited score is no longer a
hand-written weighted formula. We train a local resume-job matching classifier
on frozen DistilBERT [CLS] embeddings, then audit that learned classifier.

Outputs live in outputs_race_trained/.
"""

import argparse
import json
import random
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from transformers import AutoModel, AutoTokenizer


CATEGORY_KEYWORDS = {
    "INFORMATION-TECHNOLOGY": ["software", "engineer", "developer", "data", "python", "machine learning", "sql", "cloud"],
    "ENGINEERING": ["engineer", "mechanical", "electrical", "civil", "manufacturing", "design"],
    "FINANCE": ["finance", "financial", "accountant", "analyst", "accounting", "audit"],
    "HR": ["human resources", "hr", "recruiter", "talent", "benefits"],
    "SALES": ["sales", "account executive", "business development", "customer", "revenue"],
    "HEALTHCARE": ["nurse", "health", "clinical", "medical", "therapist", "patient"],
    "TEACHER": ["teacher", "education", "school", "instructor", "curriculum"],
    "BUSINESS-DEVELOPMENT": ["business development", "partnership", "strategy", "growth", "market"],
}

SKILLS = [
    "python", "sql", "excel", "sales", "marketing", "recruiting", "finance", "accounting",
    "customer", "project", "management", "analysis", "data", "machine", "learning",
    "communication", "leadership", "health", "clinical", "education", "design", "cloud",
]

# Racialized-name audit sets. Keep train/test names disjoint for the probe.
BLACK_TRAIN_NAMES = ["Jamal", "Lakisha", "Darnell", "Latoya", "Tyrone", "Aisha"]
WHITE_TRAIN_NAMES = ["Brad", "Emily", "Greg", "Anne", "Todd", "Meredith"]
BLACK_TEST_NAMES = ["Malik", "Keisha", "DeShawn", "Shanice"]
WHITE_TEST_NAMES = ["Brendan", "Allison", "Geoffrey", "Claire"]


def clean_text(value, max_chars):
    text = "" if pd.isna(value) else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def word_set(text):
    stop = {
        "the", "and", "for", "with", "that", "this", "you", "are", "from", "job", "resume",
        "candidate", "name", "your", "our", "will", "have", "has", "was", "were", "but",
    }
    return {t for t in re.findall(r"[a-zA-Z]{3,}", text.lower()) if t not in stop}


def load_source_rows(root, n_base, seed):
    rng = random.Random(seed)
    resumes = pd.read_csv(root / "archive" / "Resume" / "Resume.csv", usecols=["ID", "Resume_str", "Category"])
    categories = list(CATEGORY_KEYWORDS)
    resumes = resumes[resumes["Category"].isin(categories)].copy()

    jobs = pd.read_csv(
        root / "postings.csv",
        usecols=["job_id", "title", "description", "formatted_experience_level", "formatted_work_type"],
        dtype=str,
    ).dropna(subset=["title", "description"])
    jobs["job_text_lower"] = (jobs["title"].fillna("") + " " + jobs["description"].fillna("")).str.lower()

    buckets = {}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        mask = False
        for kw in keywords:
            mask = mask | jobs["job_text_lower"].str.contains(re.escape(kw.lower()), na=False)
        buckets[cat] = jobs[mask].head(700).to_dict("records")

    rows = []
    per_cat = max(1, n_base // len(categories))
    for cat in categories:
        cat_resumes = resumes[resumes["Category"] == cat].sample(frac=1.0, random_state=seed)
        pos_jobs = buckets.get(cat) or jobs.head(700).to_dict("records")
        neg_cats = [c for c in categories if c != cat and buckets.get(c)]
        for _, resume in cat_resumes.head(per_cat).iterrows():
            pos_job = rng.choice(pos_jobs)
            neg_cat = rng.choice(neg_cats)
            neg_job = rng.choice(buckets[neg_cat])
            base = {
                "base_id": len(rows),
                "resume_id": str(resume["ID"]),
                "category": cat,
                "resume": clean_text(resume["Resume_str"], 900),
                "pos_job_id": str(pos_job["job_id"]),
                "pos_job": clean_text(str(pos_job["title"]) + ". " + str(pos_job["description"]), 700),
                "pos_experience_level": clean_text(pos_job.get("formatted_experience_level", ""), 50),
                "pos_work_type": clean_text(pos_job.get("formatted_work_type", ""), 50),
                "neg_job_id": str(neg_job["job_id"]),
                "neg_job": clean_text(str(neg_job["title"]) + ". " + str(neg_job["description"]), 700),
                "neg_experience_level": clean_text(neg_job.get("formatted_experience_level", ""), 50),
                "neg_work_type": clean_text(neg_job.get("formatted_work_type", ""), 50),
            }
            rows.append(base)
            if len(rows) >= n_base:
                break
        if len(rows) >= n_base:
            break
    return rows


def make_text(name, resume, job):
    return f"Candidate name: {name}. Resume: {resume} [SEP] Job description: {job}"


def build_training_rows(base_rows, seed):
    rng = random.Random(seed)
    all_names = BLACK_TRAIN_NAMES + WHITE_TRAIN_NAMES
    rows = []
    for row in base_rows:
        for label, job_key, job_id_key, exp_key, work_key in [
            (1, "pos_job", "pos_job_id", "pos_experience_level", "pos_work_type"),
            (0, "neg_job", "neg_job_id", "neg_experience_level", "neg_work_type"),
        ]:
            # Balance race cues across positive and negative labels.
            for race, names, race_label in [
                ("black_associated", BLACK_TRAIN_NAMES, 1),
                ("white_associated", WHITE_TRAIN_NAMES, 0),
            ]:
                name = rng.choice(names)
                rows.append(
                    {
                        "base_id": row["base_id"],
                        "resume_id": row["resume_id"],
                        "job_id": row[job_id_key],
                        "category": row["category"],
                        "resume": row["resume"],
                        "job": row[job_key],
                        "experience_level": row[exp_key],
                        "work_type": row[work_key],
                        "name": name,
                        "race_group": race,
                        "race_label": race_label,
                        "match_label": label,
                        "text": make_text(name, row["resume"], row[job_key]),
                    }
                )
    return pd.DataFrame(rows)


def build_race_counterfactual_rows(base_rows, split_name, names_black, names_white):
    rows = []
    for row in base_rows:
        for name_b, name_w in zip(names_black, names_white):
            for race, name, race_label in [
                ("black_associated", name_b, 1),
                ("white_associated", name_w, 0),
            ]:
                rows.append(
                    {
                        "base_id": row["base_id"],
                        "resume_id": row["resume_id"],
                        "job_id": row["pos_job_id"],
                        "category": row["category"],
                        "resume": row["resume"],
                        "job": row["pos_job"],
                        "experience_level": row["pos_experience_level"],
                        "work_type": row["pos_work_type"],
                        "name": name,
                        "race_group": race,
                        "race_label": race_label,
                        "name_split": split_name,
                        "match_label": 1,
                        "text": make_text(name, row["resume"], row["pos_job"]),
                    }
                )
    return pd.DataFrame(rows)


def embed_texts(texts, tokenizer, model, device, batch_size, return_hidden=False, max_length=256):
    pooled = []
    hidden_by_layer = None
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            enc = tokenizer(
                texts[start:start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            pooled.append(out.last_hidden_state[:, 0, :].detach().cpu().numpy())
            if return_hidden:
                cls_layers = [h[:, 0, :].detach().cpu().numpy() for h in out.hidden_states]
                if hidden_by_layer is None:
                    hidden_by_layer = [[] for _ in cls_layers]
                for i, arr in enumerate(cls_layers):
                    hidden_by_layer[i].append(arr)
    pooled = np.vstack(pooled)
    if return_hidden:
        return pooled, [np.vstack(chunks) for chunks in hidden_by_layer]
    return pooled


def cosine_rows(a, b):
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    denom = np.where(denom == 0, 1.0, denom)
    return (np.sum(a * b, axis=1) / denom + 1.0) / 2.0


def extract_features(df, semantic_score):
    records = []
    for idx, row in df.iterrows():
        resume_words = word_set(row["resume"])
        job_words = word_set(row["job"])
        resume_lower = row["resume"].lower()
        job_lower = row["job"].lower()
        skill_resume = {s for s in SKILLS if s in resume_lower}
        skill_job = {s for s in SKILLS if s in job_lower}
        category_keywords = CATEGORY_KEYWORDS.get(row["category"], [])
        category_keyword_hits = sum(1 for kw in category_keywords if kw.lower() in job_lower)
        records.append(
            {
                "semantic_score": float(semantic_score[idx]),
                "category_job_keyword_hits": category_keyword_hits,
                "category_job_keyword_ratio": category_keyword_hits / max(1, len(category_keywords)),
                "skill_overlap": len(skill_resume & skill_job),
                "skill_overlap_ratio": len(skill_resume & skill_job) / max(1, len(skill_job)),
                "resume_job_token_overlap": len(resume_words & job_words),
                "resume_job_token_overlap_ratio": len(resume_words & job_words) / max(1, len(job_words)),
                "resume_length_words": len(resume_words),
                "job_length_words": len(job_words),
                "has_degree": int(bool(re.search(r"\b(degree|bachelor|master|phd|university|college)\b", resume_lower))),
                "has_management": int("management" in resume_lower or "manager" in resume_lower),
                "years_mentions": len(re.findall(r"\b\d+\+?\s+years?\b", resume_lower)),
                "race_cue_black_associated": int(row["race_label"] == 1),
                "name_length": len(row["name"]),
                "experience_mid_senior": int("mid" in str(row["experience_level"]).lower() or "senior" in str(row["experience_level"]).lower()),
                "work_full_time": int("full" in str(row["work_type"]).lower()),
            }
        )
    return pd.DataFrame(records)


def train_matching_head(x, y, groups, seed):
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, test_idx = next(splitter.split(x, y, groups))
    clf = HistGradientBoostingClassifier(
        max_iter=160,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=0.01,
        random_state=seed,
    )
    clf.fit(x[train_idx], y[train_idx])
    pred_prob = clf.predict_proba(x[test_idx])[:, 1]
    metrics = {
        "match_head_test_auc": float(roc_auc_score(y[test_idx], pred_prob)),
        "match_head_test_accuracy": float(accuracy_score(y[test_idx], pred_prob >= 0.5)),
    }
    return clf, metrics


def semantic_scores_for_df(df, text_emb, tokenizer, model, device, batch_size):
    unique_jobs = df[["job_id", "job"]].drop_duplicates("job_id").reset_index(drop=True)
    unique_job_emb = embed_texts(unique_jobs["job"].tolist(), tokenizer, model, device, batch_size)
    job_emb_map = {job_id: unique_job_emb[i] for i, job_id in enumerate(unique_jobs["job_id"])}
    job_emb = np.vstack([job_emb_map[job_id] for job_id in df["job_id"]])
    return cosine_rows(text_emb, job_emb)


def probe_heldout_names(hidden_train, y_train, hidden_test, y_test):
    rows = []
    rng = np.random.default_rng(7)
    for layer in range(len(hidden_train)):
        clf = LogisticRegression(max_iter=300, solver="liblinear")
        clf.fit(hidden_train[layer], y_train)
        prob = clf.predict_proba(hidden_test[layer])[:, 1]
        raw_auc = roc_auc_score(y_test, prob)
        shuffled = y_train.copy()
        rng.shuffle(shuffled)
        neg = LogisticRegression(max_iter=300, solver="liblinear")
        neg.fit(hidden_train[layer], shuffled)
        neg_prob = neg.predict_proba(hidden_test[layer])[:, 1]
        neg_auc = roc_auc_score(y_test, neg_prob)
        rows.append(
            {
                "layer": layer,
                "heldout_name_raw_auc": float(raw_auc),
                "heldout_name_decodability_auc": float(max(raw_auc, 1.0 - raw_auc)),
                "shuffled_label_control_auc": float(max(neg_auc, 1.0 - neg_auc)),
            }
        )
    return pd.DataFrame(rows)


def race_behavior_gap(df):
    rows = []
    for base_id, group in df.groupby("base_id"):
        black = group[group["race_label"] == 1]["local_model_score"].mean()
        white = group[group["race_label"] == 0]["local_model_score"].mean()
        same_black = group[group["race_label"] == 1]["local_model_score"].max() - group[group["race_label"] == 1]["local_model_score"].min()
        same_white = group[group["race_label"] == 0]["local_model_score"].max() - group[group["race_label"] == 0]["local_model_score"].min()
        rows.append(
            {
                "base_id": base_id,
                "black_associated_score": black,
                "white_associated_score": white,
                "signed_gap_black_minus_white": black - white,
                "abs_gap": abs(black - white),
                "same_race_black_name_range": same_black,
                "same_race_white_name_range": same_white,
            }
        )
    return pd.DataFrame(rows)


def plot_behavior(df, path):
    plt.figure(figsize=(7.2, 4.2))
    plt.hist(df["signed_gap_black_minus_white"], bins=22, color="#2f6f8f", alpha=0.85)
    plt.axvline(df["signed_gap_black_minus_white"].mean(), color="#c94c4c", linewidth=2, label=f"mean = {df['signed_gap_black_minus_white'].mean():.4f}")
    plt.axvline(0, color="#555555", linestyle="--", linewidth=1)
    plt.xlabel("score(Black-associated names) - score(White-associated names)")
    plt.ylabel("Resume-job pairs")
    plt.title("Racialized-Name Behavioral Counterfactual Gap")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_probe(df, path):
    plt.figure(figsize=(7.2, 4.2))
    plt.plot(df["layer"], df["heldout_name_decodability_auc"], marker="o", linewidth=2, color="#7b4ab2", label="Held-out names")
    plt.plot(df["layer"], df["shuffled_label_control_auc"], marker="s", linewidth=1.5, color="#888888", label="Shuffled-label control")
    plt.axhline(0.5, color="#555555", linestyle="--", linewidth=1)
    plt.ylim(0.45, 1.0)
    plt.xticks(df["layer"])
    plt.xlabel("DistilBERT layer (0 = embedding, 6 = final)")
    plt.ylabel("Race-associated name decodability AUC")
    plt.title("Activation Probe With Held-Out Names")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_shap_summary(shap_values, x_test, path):
    plt.figure()
    shap.summary_plot(shap_values, x_test, show=False, max_display=12)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_local_shap(shap_values, x_test, path):
    i = int(np.argmax(np.abs(shap_values.values).sum(axis=1)))
    vals = shap_values.values[i]
    order = np.argsort(np.abs(vals))[-9:]
    colors = ["#2d8a5f" if v >= 0 else "#c94c4c" for v in vals[order]]
    plt.figure(figsize=(7.4, 4.3))
    plt.barh(x_test.columns[order], vals[order], color=colors)
    plt.axvline(0, color="#555555", linewidth=1)
    plt.xlabel("SHAP contribution to XGBoost surrogate score")
    plt.title("Local SHAP Explanation")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    return i


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--n-base", type=int, default=220)
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out = root / "outputs_race_trained"
    out.mkdir(exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    base_rows = load_source_rows(root, args.n_base, args.seed)
    train_df = build_training_rows(base_rows, args.seed)
    cf_train_df = build_race_counterfactual_rows(base_rows, "probe_train_names", BLACK_TRAIN_NAMES, WHITE_TRAIN_NAMES)
    cf_test_df = build_race_counterfactual_rows(base_rows, "heldout_audit_names", BLACK_TEST_NAMES, WHITE_TEST_NAMES)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModel.from_pretrained(args.model, output_hidden_states=True, local_files_only=True)
    model.to(device)
    model.eval()

    train_text_emb = embed_texts(train_df["text"].tolist(), tokenizer, model, device, args.batch_size)
    train_semantic = semantic_scores_for_df(train_df, train_text_emb, tokenizer, model, device, args.batch_size)
    train_features = extract_features(train_df.reset_index(drop=True), train_semantic)
    match_head, match_metrics = train_matching_head(
        train_features.to_numpy(),
        train_df["match_label"].to_numpy(),
        train_df["base_id"].to_numpy(),
        args.seed,
    )

    cf_train_x, cf_train_hidden = embed_texts(
        cf_train_df["text"].tolist(), tokenizer, model, device, args.batch_size, return_hidden=True
    )
    cf_test_x, cf_test_hidden = embed_texts(
        cf_test_df["text"].tolist(), tokenizer, model, device, args.batch_size, return_hidden=True
    )
    # Semantic feature for both the trained local matcher and XGBoost surrogate.
    # Unlike the earlier script, this feature is explicitly included downstream.
    semantic_score = semantic_scores_for_df(cf_test_df, cf_test_x, tokenizer, model, device, args.batch_size)
    features = extract_features(cf_test_df.reset_index(drop=True), semantic_score)
    cf_test_df["local_model_score"] = match_head.predict_proba(features.to_numpy())[:, 1]
    target = cf_test_df["local_model_score"].to_numpy()

    tr_idx, te_idx = train_test_split(
        np.arange(len(features)),
        test_size=0.25,
        random_state=args.seed,
        stratify=cf_test_df["race_label"].to_numpy(),
    )
    x_train, x_test = features.iloc[tr_idx], features.iloc[te_idx]
    y_train, y_test = target[tr_idx], target[te_idx]
    surrogate = xgb.XGBRegressor(
        n_estimators=450,
        max_depth=4,
        learning_rate=0.035,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=args.seed,
    )
    surrogate.fit(x_train, y_train)
    pred = surrogate.predict(x_test)

    explainer = shap.Explainer(surrogate, x_train)
    shap_values = explainer(x_test)
    shap_importance = pd.DataFrame(
        {
            "feature": x_test.columns,
            "mean_abs_shap": np.abs(shap_values.values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)

    probe_df = probe_heldout_names(
        cf_train_hidden,
        cf_train_df["race_label"].to_numpy(),
        cf_test_hidden,
        cf_test_df["race_label"].to_numpy(),
    )
    gap_df = race_behavior_gap(cf_test_df)

    cf_test_df.to_csv(out / "scored_racial_counterfactuals.csv", index=False)
    train_df.to_csv(out / "matching_training_rows.csv", index=False)
    features.assign(local_model_score=target).to_csv(out / "surrogate_features.csv", index=False)
    gap_df.to_csv(out / "racial_behavioral_gap.csv", index=False)
    probe_df.to_csv(out / "heldout_name_probe_auc.csv", index=False)
    shap_importance.to_csv(out / "xgboost_shap_importance.csv", index=False)

    plot_behavior(gap_df, out / "racial_behavioral_gap.png")
    plot_probe(probe_df, out / "heldout_name_probe_auc.png")
    plot_shap_summary(shap_values, x_test, out / "xgboost_global_shap.png")
    local_row = plot_local_shap(shap_values, x_test, out / "xgboost_local_shap.png")

    metrics = {
        "model": args.model,
        "device": device,
        "n_base_resume_job_pairs": len(base_rows),
        "matching_training_rows": int(len(train_df)),
        "racial_counterfactual_audit_rows": int(len(cf_test_df)),
        **match_metrics,
        "racial_behavioral_gap_mean_abs": float(gap_df["abs_gap"].mean()),
        "racial_behavioral_gap_mean_signed_black_minus_white": float(gap_df["signed_gap_black_minus_white"].mean()),
        "racial_behavioral_gap_max_abs": float(gap_df["abs_gap"].max()),
        "same_race_black_name_range_mean": float(gap_df["same_race_black_name_range"].mean()),
        "same_race_white_name_range_mean": float(gap_df["same_race_white_name_range"].mean()),
        "probe_heldout_auc_layer_0": float(probe_df.loc[probe_df["layer"] == 0, "heldout_name_decodability_auc"].iloc[0]),
        "probe_heldout_auc_final_layer": float(probe_df["heldout_name_decodability_auc"].iloc[-1]),
        "probe_heldout_auc_max": float(probe_df["heldout_name_decodability_auc"].max()),
        "probe_shuffled_control_final_layer": float(probe_df["shuffled_label_control_auc"].iloc[-1]),
        "xgboost_surrogate_fidelity_r2": float(r2_score(y_test, pred)),
        "xgboost_surrogate_fidelity_mae": float(mean_absolute_error(y_test, pred)),
        "race_cue_black_mean_abs_shap": float(
            shap_importance.loc[
                shap_importance["feature"] == "race_cue_black_associated", "mean_abs_shap"
            ].iloc[0]
        ),
        "top_shap_features": shap_importance.head(10).to_dict("records"),
        "local_shap_test_row_index": local_row,
        "outputs": [
            "outputs_race_trained/racial_behavioral_gap.png",
            "outputs_race_trained/heldout_name_probe_auc.png",
            "outputs_race_trained/xgboost_global_shap.png",
            "outputs_race_trained/xgboost_local_shap.png",
            "outputs_race_trained/audit_metrics.json",
        ],
    }
    (out / "audit_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
