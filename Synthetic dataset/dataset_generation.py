# Engifi synthetic dataset (task-level) generator — compact version
# Schema: person_id, role, task, [Psychological + Cognitive inputs], Predictions: Performance_task, Satisfaction_task, Fit_task
import numpy as np
import pandas as pd
from numpy.random import default_rng

rng = default_rng(7)

# -------- Feature sets
COG = ["Gf","Gc","WM","Speed"]
PSY = ["Conscientiousness","Agreeableness","AutonomyPref"]
RIASEC = ["R","I","A","S","E","C"]
VALUES = ["Value_Growth","Value_Stability","Value_CustomerImpact"]
ALL_INPUTS = COG + PSY + RIASEC + VALUES

# -------- Roles (subset from tech/engineering) with means (H/M/L = +0.6 / 0 / -0.6)
H, M, L = 0.6, 0.0, -0.6
ROLES = {
    "Backend Engineer":       {"means": {"Gf":H,"Gc":M,"WM":H,"Speed":M,"Conscientiousness":H,"Agreeableness":M,"AutonomyPref":0.2,
                                         "R":0.1,"I":0.6,"A":-0.1,"S":0.0,"E":-0.1,"C":0.2,
                                         "Value_Growth":0.3,"Value_Stability":0.1,"Value_CustomerImpact":0.1},
                               "autonomy":0.2,
                               "perf_w":{"Gf":.22,"WM":.18,"Speed":.12,"Gc":.10,"Conscientiousness":.18} },
    "Frontend Engineer":      {"means": {"Gf":M,"Gc":M,"WM":M,"Speed":H,"Conscientiousness":M,"Agreeableness":M,"AutonomyPref":0.3,
                                         "R":-0.1,"I":0.3,"A":0.5,"S":0.2,"E":0.0,"C":0.0,
                                         "Value_Growth":0.2,"Value_Stability":0.0,"Value_CustomerImpact":0.4},
                               "autonomy":0.3,
                               "perf_w":{"Speed":.20,"WM":.14,"Gf":.12,"Gc":.10,"Conscientiousness":.14} },
    "DevOps/SRE":             {"means": {"Gf":H,"Gc":M,"WM":H,"Speed":H,"Conscientiousness":H,"Agreeableness":M,"AutonomyPref":0.1,
                                         "R":0.2,"I":0.5,"A":-0.1,"S":0.1,"E":-0.1,"C":0.3,
                                         "Value_Growth":0.1,"Value_Stability":0.4,"Value_CustomerImpact":0.1},
                               "autonomy":0.1,
                               "perf_w":{"Speed":.18,"WM":.18,"Gf":.16,"Conscientiousness":.16,"Gc":.10} },
    "Security Engineer":      {"means": {"Gf":H,"Gc":M,"WM":H,"Speed":M,"Conscientiousness":H,"Agreeableness":L,"AutonomyPref":0.0,
                                         "R":0.1,"I":0.6,"A":-0.2,"S":0.0,"E":-0.1,"C":0.3,
                                         "Value_Growth":0.0,"Value_Stability":0.4,"Value_CustomerImpact":0.1},
                               "autonomy":0.0,
                               "perf_w":{"Gf":.20,"WM":.18,"Conscientiousness":.16,"Speed":.12,"Gc":.10} },
    "Data Scientist":         {"means": {"Gf":H,"Gc":0.3,"WM":H,"Speed":M,"Conscientiousness":M,"Agreeableness":M,"AutonomyPref":0.3,
                                         "R":0.0,"I":0.7,"A":0.1,"S":0.0,"E":0.0,"C":0.2,
                                         "Value_Growth":0.4,"Value_Stability":0.0,"Value_CustomerImpact":0.1},
                               "autonomy":0.3,
                               "perf_w":{"Gf":.22,"WM":.18,"Gc":.14,"Speed":.10,"Conscientiousness":.12} },
    "ML Engineer":            {"means": {"Gf":H,"Gc":M,"WM":H,"Speed":H,"Conscientiousness":H,"Agreeableness":M,"AutonomyPref":0.2,
                                         "R":0.1,"I":0.6,"A":-0.1,"S":0.0,"E":0.0,"C":0.3,
                                         "Value_Growth":0.2,"Value_Stability":0.2,"Value_CustomerImpact":0.2},
                               "autonomy":0.2,
                               "perf_w":{"Speed":.18,"WM":.18,"Gf":.16,"Conscientiousness":.16,"Gc":.10} },
    "Data Engineer":          {"means": {"Gf":0.3,"Gc":M,"WM":H,"Speed":0.3,"Conscientiousness":H,"Agreeableness":M,"AutonomyPref":0.0,
                                         "R":0.1,"I":0.5,"A":-0.1,"S":0.0,"E":0.0,"C":0.4,
                                         "Value_Growth":0.1,"Value_Stability":0.4,"Value_CustomerImpact":0.1},
                               "autonomy":0.0,
                               "perf_w":{"WM":.18,"Conscientiousness":.18,"Speed":.14,"Gf":.12,"Gc":.10} },
    "Simulation Engineer":    {"means": {"Gf":H,"Gc":M,"WM":H,"Speed":M,"Conscientiousness":H,"Agreeableness":M,"AutonomyPref":0.0,
                                         "R":0.2,"I":0.6,"A":-0.1,"S":0.0,"E":-0.1,"C":0.3,
                                         "Value_Growth":0.2,"Value_Stability":0.2,"Value_CustomerImpact":0.1},
                               "autonomy":0.0,
                               "perf_w":{"Gf":.20,"WM":.18,"Conscientiousness":.16,"Speed":.12,"Gc":.10} },
    "Manufacturing Engineer": {"means": {"Gf":0.3,"Gc":M,"WM":0.3,"Speed":H,"Conscientiousness":H,"Agreeableness":M,"AutonomyPref":-0.2,
                                         "R":0.4,"I":0.2,"A":-0.2,"S":0.0,"E":0.0,"C":0.5,
                                         "Value_Growth":0.0,"Value_Stability":0.6,"Value_CustomerImpact":0.1},
                               "autonomy":-0.2,
                               "perf_w":{"Speed":.18,"Conscientiousness":.18,"WM":.14,"Gf":.12,"Gc":.10} },
    "Technical Writer":       {"means": {"Gf":M,"Gc":0.3,"WM":0.3,"Speed":M,"Conscientiousness":H,"Agreeableness":0.3,"AutonomyPref":-0.1,
                                         "R":-0.1,"I":0.3,"A":0.4,"S":0.2,"E":0.0,"C":0.4,
                                         "Value_Growth":0.1,"Value_Stability":0.4,"Value_CustomerImpact":0.2},
                               "autonomy":-0.1,
                               "perf_w":{"Conscientiousness":.20,"WM":.14,"Gc":.12,"Speed":.10,"Gf":.10} },
}

# -------- Tasks per role with demand modifiers (tilt perf toward certain traits)
# Each task has a dict of modifiers that add to performance via dot with features
TASKS = {
    "Backend Engineer": [
        ("API Design",        {"Gf":.10, "WM":.06, "Conscientiousness":.06}),
        ("Database Modeling", {"Gf":.08, "Gc":.06, "Conscientiousness":.06}),
        ("Code Review",       {"Gc":.06, "Conscientiousness":.08}),
        ("Incident Fix",      {"Speed":.10, "WM":.06})
    ],
    "Frontend Engineer": [
        ("UI Implementation", {"Speed":.10,"WM":.06,"Conscientiousness":.06}),
        ("Accessibility",     {"Gc":.06,"Conscientiousness":.08}),
        ("Perf Tuning",       {"Speed":.12,"Gf":.06})
    ],
    "DevOps/SRE": [
        ("Infra as Code",     {"Conscientiousness":.10,"WM":.08}),
        ("Monitoring",        {"Speed":.10,"WM":.06}),
        ("Incident Response", {"Speed":.12,"Gf":.06})
    ],
    "Security Engineer": [
        ("Threat Modeling",   {"Gf":.12,"WM":.06}),
        ("Detection Tuning",  {"WM":.10,"Speed":.06}),
        ("Compliance Audit",  {"Gc":.08,"Conscientiousness":.10})
    ],
    "Data Scientist": [
        ("EDA",               {"Gf":.10,"WM":.08}),
        ("Modeling",          {"Gf":.12,"WM":.08}),
        ("Experimentation",   {"Gf":.08,"Conscientiousness":.06})
    ],
    "ML Engineer": [
        ("Feature Store",     {"WM":.10,"Conscientiousness":.08}),
        ("Model Serving",     {"Speed":.12,"WM":.06}),
        ("Monitoring/Drift",  {"Speed":.10,"Gf":.06})
    ],
    "Data Engineer": [
        ("Pipelines",         {"WM":.10,"Conscientiousness":.10}),
        ("Data Quality",      {"Conscientiousness":.12,"Gc":.06}),
        ("Orchestration",     {"WM":.08,"Speed":.08})
    ],
    "Simulation Engineer": [
        ("Meshing",           {"Conscientiousness":.10,"WM":.08}),
        ("Solver Tuning",     {"Gf":.10,"WM":.08}),
        ("Validation",        {"Gc":.08,"Conscientiousness":.10})
    ],
    "Manufacturing Engineer": [
        ("Line Balancing",    {"Speed":.12,"Gf":.06}),
        ("SPC/Quality",       {"Conscientiousness":.12,"WM":.06}),
        ("SOPs",              {"Conscientiousness":.14,"Gc":.06})
    ],
    "Technical Writer": [
        ("Authoring",         {"Conscientiousness":.12,"Gc":.08}),
        ("API Reference",     {"Gc":.10,"WM":.08}),
        ("Diagrams",          {"WM":.08,"Gf":.06})
    ]
}

# -------- Correlation and SDs
DEFAULT_SD = 0.6
corr = np.eye(len(ALL_INPUTS))

def set_block(names, r):
    idx = [ALL_INPUTS.index(n) for n in names]
    for i in idx:
        for j in idx:
            if i!=j: corr[i,j]=r

set_block(COG, .45)
set_block(["R","I"], .15); set_block(["I","A"], .15); set_block(["A","S"], .15); set_block(["S","E"], .15); set_block(["E","C"], .15); set_block(["R","C"], .10)
set_block(VALUES, .20)
# cognitive ↔ conscientiousness
for c in COG:
    corr[ALL_INPUTS.index(c), ALL_INPUTS.index("Conscientiousness")] = 0.10
    corr[ALL_INPUTS.index("Conscientiousness"), ALL_INPUTS.index(c)] = 0.10

# PSD tweak
eig = np.linalg.eigvalsh(corr)
if eig.min()<1e-6:
    corr += np.eye(len(ALL_INPUTS))*(1e-6 - eig.min())

ORG_VALUES = {"Value_Growth":0.30,"Value_Stability":0.10,"Value_CustomerImpact":0.40}
NOISE_PERF, NOISE_FIT, NOISE_SAT = 0.35, 0.30, 0.30

def sample_people(role, n=300):
    means_vec = np.array([ROLES[role]["means"].get(k,0.0) for k in ALL_INPUTS])
    sds_vec = np.array([DEFAULT_SD for _ in ALL_INPUTS])
    cov = corr * np.outer(sds_vec, sds_vec)
    X = rng.multivariate_normal(mean=means_vec, cov=cov, size=n)
    df = pd.DataFrame(X, columns=ALL_INPUTS)
    df["role"] = role
    return df

def derive_gma(df):
    X = df[COG].to_numpy()
    Xz = (X - X.mean(axis=0))/X.std(axis=0, ddof=0)
    C = np.cov(Xz, rowvar=False)
    w = np.linalg.eigh(C)[1][:, -1]
    g = Xz @ w
    g = (g - g.mean())/g.std(ddof=0)
    return g

rows = []
person_counter = 0
for role, spec in ROLES.items():
    people = sample_people(role, n=400)  # 400 per role
    people["person_id"] = np.arange(person_counter, person_counter+len(people))
    person_counter += len(people)
    people["GMA"] = derive_gma(people)

    # Precompute fit components (role-level)
    # Interests cosine vs role interest intent (based on means)
    role_intent = np.array([spec["means"].get(k,0.0) for k in RIASEC])
    norm = np.linalg.norm(role_intent)
    if norm==0: role_intent = np.ones(6)*1e-6; norm=np.linalg.norm(role_intent)
    role_intent = role_intent / norm

    # Values org match
    org_vec = np.array([ORG_VALUES[v] for v in VALUES])
    # Autonomy
    role_autonomy = spec["autonomy"]

    for task_name, tmods in TASKS[role]:
        df = people.copy()
        df["task"] = task_name

        # Base performance: role perf weights dot inputs
        perf = np.zeros(len(df))
        for k,w in spec["perf_w"].items():
            perf += w * df[k].to_numpy()
        # Task demand tilt
        for k,w in tmods.items():
            perf += w * df[k].to_numpy()
        perf += rng.normal(0, NOISE_PERF, size=len(df))

        # Fit: interests + values + autonomy
        P = df[RIASEC].to_numpy()
        Pn = P / np.clip(np.linalg.norm(P, axis=1, keepdims=True), 1e-6, None)
        interest_match = (Pn @ role_intent.reshape(-1,1)).flatten()  # [-1,1]
        V = df[VALUES].to_numpy()
        values_match = -np.linalg.norm(V - org_vec, axis=1)          # higher is better
        auto_penalty = -np.abs(df["AutonomyPref"].to_numpy() - role_autonomy)

        fit = 0.55*interest_match + 0.25*values_match + 0.20*auto_penalty + rng.normal(0, NOISE_FIT, size=len(df))

        # Satisfaction: from fit + standardized performance + values
        perf_z = (perf - perf.mean())/(perf.std(ddof=0) if perf.std(ddof=0)>0 else 1.0)
        sat = 0.48*fit + 0.22*perf_z + 0.15*values_match + rng.normal(0, NOISE_SAT, size=len(df))

        out = df[["person_id","role","task"] + (PSY + COG + ["GMA"]) + RIASEC + VALUES].copy()
        out["Performance_task"] = perf
        out["Fit_task"] = fit
        out["Satisfaction_task"] = sat
        rows.append(out)

dataset = pd.concat(rows, ignore_index=True)

# Save and preview
path = "/mnt/data/engifi_task_level_synth.csv"
dataset.to_csv(path, index=False)

import caas_jupyter_tools
caas_jupyter_tools.display_dataframe_to_user("Engifi Task-Level Synthetic Dataset (preview)", dataset.head(30))

path
