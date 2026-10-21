"""Kiểm tra độc lập Bước 0-6: tính lại từ dữ liệu gốc bằng sklearn/pandas, đối chiếu bảng kết quả.

Cách dùng (từ gốc repo, sau khi đã chạy xong Bước 0-6):
    python scripts/audit_steps_0_6.py .

Không import code tính độ đo của dự án (chỉ dùng load_graphs để đọc graphs.pt).
In ra PASS/FAIL cho từng mục.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(sys.argv[1])
EXP = ROOT / "experiments" / "official"
TAB = ROOT / "results" / "tables" / "official"
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(("PASS " if ok else "FAIL ") + name + (f" | {detail}" if detail else ""))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------- Bước 1-2: số liệu thô ----------------
raw = ROOT / "data" / "raw" / "2" / "27"
ev_files = [p for p in raw.glob("*.json") if not p.name.endswith(".lineups.json")]
lu_files = list(raw.glob("*.lineups.json"))
check(
    "B1: 380 file sự kiện + 380 file đội hình",
    len(ev_files) == 380 and len(lu_files) == 380,
    f"{len(ev_files)} / {len(lu_files)}",
)
n_events = sum(len(json.loads(p.read_text(encoding="utf-8"))) for p in ev_files)
coll = pd.read_csv(TAB / "collection_report.csv").iloc[0]
check("B1: tổng sự kiện đếm lại = collection_report", n_events == coll.events_total, f"{n_events}")

poss = pd.read_parquet(ROOT / "data/interim/official/possessions.parquet")
pev = pd.read_parquet(ROOT / "data/interim/official/possession_events.parquet")
summ = pd.read_csv(TAB / "data_pipeline_summary.csv").iloc[0]
check("B2: số possession sạch", len(poss) == summ.n_clean_possessions, f"{len(poss)}")
check("B2: số đường chuyền", len(pev) == summ.n_successful_passes, f"{len(pev)}")
check(
    "B2: số possession có sút",
    int(poss.has_shot.sum()) == summ.shot_positive_count,
    f"{int(poss.has_shot.sum())}",
)
check(
    "B2: khóa (match_id, possession_id) duy nhất",
    not poss.duplicated(["match_id", "possession_id"]).any(),
)
check("B2: mọi possession >= 2 đường chuyền", (poss.n_passes >= 2).all())
check("B2: chỉ periods 1-2", set(poss.period.unique()) <= {1, 2})
check(
    "B2: đường chuyền đủ tọa độ + người nhận",
    pev[["start_x", "start_y", "end_x", "end_y", "recipient_id", "passer_id"]].notna().all().all(),
)
pt = pev.merge(
    poss[["match_id", "possession_id", "team_id"]],
    on=["match_id", "possession_id"],
    suffixes=("", "_poss"),
)
check("B2: mọi đường chuyền thuộc đội kiểm soát bóng", (pt.team_id == pt.team_id_poss).all())
cnt = pev.groupby(["match_id", "possession_id"]).size().rename("n").reset_index()
m = poss.merge(cnt, on=["match_id", "possession_id"])
check("B2: n_passes khớp số dòng đường chuyền", (m.n == m.n_passes).all())

# ---------------- Bước 3: đồ thị ----------------
sys.path.insert(0, str(ROOT / "src"))
from xai_football.graphs.dataset import load_graphs  # noqa: E402

graphs, gidx, names = load_graphs(ROOT / "data/processed/official")
gb = pd.read_csv(TAB / "graph_build_report.csv").iloc[0]
check(
    "B3: số đồ thị = 45047 và khớp report",
    len(graphs) == gb.graphs_output == 45047,
    f"{len(graphs)}",
)
check("B3: số đồ thị dương khớp report", int(sum(g.y.item() for g in graphs)) == gb.positive_graphs)
lab = poss.set_index(["match_id", "possession_id"])
ok_label = all(
    int(g.y.item()) == int(lab.loc[(g.match_id, g.possession_id), "has_shot"]) for g in graphs
)
check("B3: nhãn đồ thị = has_shot của possession", ok_label)
fsi = lab["first_shot_index"]
pev_idx = pev.set_index("event_id")["event_index"]
after = 0
for g in graphs:
    fs = fsi.loc[(g.match_id, g.possession_id)]
    if pd.notna(fs):
        after += int((pev_idx.loc[g.edge_event_ids].to_numpy() >= fs).sum())
check("B3: không cạnh nào sau cú sút đầu tiên (tính lại)", after == 0, f"{after}")
xs = np.concatenate([g.edge_attr[:, :5].numpy().ravel() for g in graphs])
check(
    "B3: tọa độ/độ dài chuẩn hóa nằm trong [0,1]",
    xs.min() >= 0 and xs.max() <= 1,
    f"min {xs.min():.4f} max {xs.max():.4f}",
)
check(
    "B3: x 8 chiều, edge_attr 12 chiều",
    all(g.x.shape[1] == 8 and g.edge_attr.shape[1] == 12 for g in graphs),
)

# ---------------- Bước 4: chia tập ----------------
sp = {
    s: set(pd.read_csv(ROOT / f"data/splits/{s}_matches.csv").match_id)
    for s in ("train", "val", "test")
}
check(
    "B4: 266/57/57 trận, không trùng, phủ 380",
    [len(sp[s]) for s in ("train", "val", "test")] == [266, 57, 57]
    and not (sp["train"] & sp["val"])
    and not (sp["train"] & sp["test"])
    and not (sp["val"] & sp["test"])
    and len(sp["train"] | sp["val"] | sp["test"]) == 380 == poss.match_id.nunique(),
)
test_graphs = [g for g in graphs if g.match_id in sp["test"]]
val_graphs = [g for g in graphs if g.match_id in sp["val"]]
gss = pd.read_csv(TAB / "graph_split_summary.csv").set_index("split")
check(
    "B4: số đồ thị test/val khớp bảng",
    len(test_graphs) == gss.loc["test", "n_graphs"]
    and len(val_graphs) == gss.loc["val", "n_graphs"],
    f"test {len(test_graphs)} val {len(val_graphs)}",
)
test_keys = pd.DataFrame(
    {
        "match_id": [g.match_id for g in test_graphs],
        "possession_id": [g.possession_id for g in test_graphs],
        "y": [int(g.y.item()) for g in test_graphs],
    }
)
val_keys = pd.DataFrame(
    {
        "match_id": [g.match_id for g in val_graphs],
        "possession_id": [g.possession_id for g in val_graphs],
        "y": [int(g.y.item()) for g in val_graphs],
    }
)


# ---------------- Bước 5: artifact huấn luyện ----------------
def best_f1_threshold(y, p):
    best, th_best = -1, 0.5
    for th in np.linspace(0.05, 0.95, 100):
        s = f1_score(y, (p >= th).astype(int), zero_division=0)
        if s > best:
            best, th_best = s, th
    return round(float(th_best), 4)


def metrics(y, p, th):
    yp = (p >= th).astype(int)
    return {
        "accuracy": accuracy_score(y, yp),
        "precision": precision_score(y, yp, zero_division=0),
        "recall": recall_score(y, yp, zero_division=0),
        "f1": f1_score(y, yp, zero_division=0),
        "auc_roc": roc_auc_score(y, p),
        "auc_pr": average_precision_score(y, p),
        "brier_score": brier_score_loss(y, p),
    }


import yaml  # noqa: E402

_train_cfg = yaml.safe_load((ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
SEEDS = [int(s) for s in _train_cfg["seeds"][: int(_train_cfg["n_runs"])]]
_base_agg = json.loads((EXP / "baselines" / "aggregate_summary.json").read_text(encoding="utf-8"))
check(
    "B5: seed của baseline = đúng danh sách seed trong configs/train.yaml, cờ complete = True",
    [int(s) for s in _base_agg["configured_seeds"]] == SEEDS and _base_agg["is_complete_official"],
    f"{SEEDS}",
)
recomputed = {}  # display -> list of per-seed metric dicts
probs_test = {}
for key, disp in (
    ("gcn", "GCN (đề xuất)"),
    ("mpnn", "MPNN (đề xuất)"),
    ("gatv2", "GATv2 (baseline)"),
    ("graphsage", "GraphSAGE (baseline)"),
):
    agg = json.loads((EXP / key / "aggregate_summary.json").read_text(encoding="utf-8"))
    check(
        f"B5 {key}: seed = đúng danh sách trong configs/train.yaml, complete, 0 lỗi",
        agg["successful_seeds"] == SEEDS
        and agg["is_complete_official"]
        and agg["n_failed_runs"] == 0,
    )
    per = []
    vals = {}
    for s in SEEDS:
        d = EXP / key / str(s)
        tp = pd.read_parquet(d / "test_predictions.parquet")
        vp = pd.read_parquet(d / "val_predictions.parquet")
        ok_keys = (
            tp[["match_id", "possession_id"]]
            .reset_index(drop=True)
            .equals(test_keys[["match_id", "possession_id"]])
            and (tp.y_true.to_numpy() == test_keys.y.to_numpy()).all()
            and vp[["match_id", "possession_id"]]
            .reset_index(drop=True)
            .equals(val_keys[["match_id", "possession_id"]])
            and (vp.y_true.to_numpy() == val_keys.y.to_numpy()).all()
        )
        th = best_f1_threshold(vp.y_true.to_numpy(), vp.y_prob.to_numpy())
        man = json.loads((d / "run_manifest.json").read_text(encoding="utf-8"))
        mt = metrics(tp.y_true.to_numpy(), tp.y_prob.to_numpy(), th)
        mv = metrics(vp.y_true.to_numpy(), vp.y_prob.to_numpy(), th)
        tm = pd.read_csv(d / "test_metrics.csv").iloc[0]
        same = abs(th - man["threshold"]) < 1e-9 and all(
            abs(round(mt[k], 4) - tm[k]) < 1e-9 for k in mt
        )
        check(
            f"B5 {key} seed {s}: khóa dự đoán đúng tập test/val; ngưỡng chọn trên VAL; độ đo test tính lại khớp",
            ok_keys and same,
            f"th={th} auc_pr={mt['auc_pr']:.4f}",
        )
        per.append(mt)
        vals[s] = mv["auc_pr"]
        probs_test.setdefault(disp, []).append(tp.y_prob.to_numpy())
    recomputed[disp] = per
    best = max(SEEDS, key=lambda s: round(vals[s], 4))
    check(
        f"B5 {key}: seed được chọn = seed có Val AUC-PR cao nhất",
        best == agg["best_seed_by_val_auc_pr"],
        f"{best}",
    )
    ck = ROOT / "models/official" / f"{key}_best.pt"
    check(
        f"B5 {key}: models/official/{key}_best.pt = checkpoint của seed {best}",
        sha(ck) == sha(EXP / key / str(best) / "best_model.pt"),
    )
    pw = json.loads((EXP / key / "42" / "run_manifest.json").read_text(encoding="utf-8"))[
        "pos_weight"
    ]
    tr = [g for g in graphs if g.match_id in sp["train"]]
    npos = sum(int(g.y.item()) for g in tr)
    check(
        f"B5 {key}: pos_weight = n_neg/n_pos của TRAIN",
        abs(pw - (len(tr) - npos) / npos) < 1e-5,
        f"{pw}",
    )

for prob_col, disp, key in (
    ("lr_prob", "Logistic Regression", "logistic_regression"),
    ("rf_prob", "Random Forest", "random_forest"),
    ("ann_prob", "ANN (MLP)", "ann"),
):
    per = []
    allm = pd.read_csv(EXP / "baselines" / "test_metrics_all_seeds.csv")
    for s in SEEDS:
        d = EXP / "baselines" / str(s)
        tp = pd.read_parquet(d / "test_predictions.parquet")
        vp = pd.read_parquet(d / "val_predictions.parquet")
        ok_keys = (
            tp[["match_id", "possession_id"]]
            .reset_index(drop=True)
            .equals(test_keys[["match_id", "possession_id"]])
            and (tp.y_true.to_numpy() == test_keys.y.to_numpy()).all()
        )
        th = best_f1_threshold(vp.y_true.to_numpy(), vp[prob_col].to_numpy())
        mt = metrics(tp.y_true.to_numpy(), tp[prob_col].to_numpy(), th)
        row = allm[(allm.model == key) & (allm.seed == s)].iloc[0]
        same = abs(th - row.threshold) < 1e-9 and all(
            abs(round(mt[k], 4) - row[k]) < 1e-9 for k in mt
        )
        check(
            f"B5 {key} seed {s}: khóa đúng; ngưỡng trên VAL; độ đo tính lại khớp",
            ok_keys and same,
            f"th={th}",
        )
        per.append(mt)
        probs_test.setdefault(disp, []).append(tp[prob_col].to_numpy())
    recomputed[disp] = per

# ---------------- Bước 6: bảng tổng hợp ----------------
mc = pd.read_csv(TAB / "model_comparison.csv").set_index("model")
check(
    "B6: model_comparison có đúng 7 mô hình, đều 5 seed & complete",
    len(mc) == 7 and (mc.n_runs == 5).all() and mc.complete.all(),
)
for disp, per in recomputed.items():
    df = pd.DataFrame(per)
    ok = all(
        abs(round(df[k].mean(), 4) - mc.loc[disp, k]) <= 1.5e-4
        and abs(round(df[k].std(ddof=1), 4) - mc.loc[disp, f"{k}_std"]) <= 1.5e-4
        for k in ("accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr", "brier_score")
    )
    check(
        f"B6 {disp}: mean/std trong bảng = tính lại từ dự đoán (sai số làm tròn <= 1.5e-4)",
        ok,
        f"auc_pr {df.auc_pr.mean():.4f}±{df.auc_pr.std(ddof=1):.4f}",
    )

ci = pd.read_csv(TAB / "model_metric_ci.csv")
for disp, plist in probs_test.items():
    for metric, fn in (
        ("auc_pr", average_precision_score),
        ("auc_roc", roc_auc_score),
        ("brier_score", brier_score_loss),
    ):
        v = np.mean([fn(test_keys.y, p) for p in plist])
        row = ci[(ci.model == disp) & (ci.metric == metric)].iloc[0]
        if not (abs(round(v, 4) - row.value) < 1e-9 and row.ci95_low <= row.value <= row.ci95_high):
            check(f"B6 CI {disp} {metric}", False, f"{v} vs {row.value}")
check(
    "B6: giá trị điểm trong model_metric_ci khớp sklearn và nằm trong KTC",
    all(r[1] for r in results if r[0].startswith("B6 CI")),
)

# Bootstrap kiểm tra ĐỘC LẬP bằng sklearn (10.000 vòng, seed bootstrap KHÁC 42) cho cặp quan trọng
# nhất MPNN − Random Forest, cả AUC-PR và AUC-ROC. Tiêu chí đặt trước khi chạy:
#   - hai cận KTC 95% lệch bản báo cáo < 0.003,
#   - KTC cùng kết luận về số 0 (cùng chứa 0, hoặc cùng không chứa 0),
#   - p thô lệch < 0.01.
sig = pd.read_csv(TAB / "significance_tests.csv")
check(
    "B6: significance_tests đủ 21 cặp x 3 độ đo",
    len(sig) == 63 and sig.groupby("metric").size().eq(21).all(),
)
rng = np.random.default_rng(2026)
y = test_keys.y.to_numpy()
mids = test_keys.match_id.to_numpy()
um = np.unique(mids)
rows = {mm: np.flatnonzero(mids == mm) for mm in um}
N_BOOT_AUDIT = 10000
diffs = {"auc_pr": [], "auc_roc": []}
for _ in range(N_BOOT_AUDIT):
    idx = np.concatenate([rows[mm] for mm in rng.choice(um, len(um))])
    for metric, fn in (("auc_pr", average_precision_score), ("auc_roc", roc_auc_score)):
        a = np.mean([fn(y[idx], p[idx]) for p in probs_test["MPNN (đề xuất)"]])
        b = np.mean([fn(y[idx], p[idx]) for p in probs_test["Random Forest"]])
        diffs[metric].append(a - b)
for metric, d in diffs.items():
    d = np.asarray(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    n = len(d)
    p_raw = min(1.0, 2 * min((np.sum(d <= 0) + 1) / (n + 1), (np.sum(d >= 0) + 1) / (n + 1)))
    r = sig[
        (sig.metric == metric)
        & (sig.model_a == "MPNN (đề xuất)")
        & (sig.model_b == "Random Forest")
    ].iloc[0]
    same_side = (lo <= 0 <= hi) == (r.ci95_low <= 0 <= r.ci95_high)
    check(
        f"B6: bootstrap độc lập sklearn 10.000 vòng (seed 2026) MPNN−RF {metric}: "
        "2 cận lệch < 0.003, cùng kết luận về số 0, p thô lệch < 0.01",
        abs(lo - r.ci95_low) < 0.003
        and abs(hi - r.ci95_high) < 0.003
        and same_side
        and abs(p_raw - r.p_value) < 0.01,
        f"độc lập [{lo:+.4f}, {hi:+.4f}] p={p_raw:.4f} | báo cáo [{r.ci95_low:+.4f}, "
        f"{r.ci95_high:+.4f}] p={r.p_value:.4f}",
    )

# Holm: p_holm >= p thô, đơn điệu theo thứ tự p thô
for metric, g in sig.groupby("metric"):
    g = g.sort_values("p_value")
    ok = (g.p_value_holm >= g.p_value - 1e-9).all() and g.p_value_holm.is_monotonic_increasing
    check(f"B6: Holm hợp lệ ({metric})", ok)

em = json.loads(
    (ROOT / "results/metadata/official/experiment_manifest.json").read_text(encoding="utf-8")
)
ok = all(
    em["models"][k]["checkpoint_sha256"] == sha(ROOT / "models/official" / f"{k}_best.pt")
    for k in ("gcn", "mpnn", "gatv2", "graphsage")
)
check("B6: SHA checkpoint trong experiment_manifest khớp file thật", ok)
check(
    "B6: manifest ghi đúng số đồ thị test/val/train",
    (em["splits"]["test_graphs"], em["splits"]["val_graphs"], em["splits"]["train_graphs"])
    == (6646, 6757, 31644),
)

n_fail = sum(1 for r in results if not r[1])
print(f"\nTỔNG: {len(results)} kiểm tra, {len(results) - n_fail} PASS, {n_fail} FAIL")
sys.exit(1 if n_fail else 0)
