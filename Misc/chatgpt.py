import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------- DATA LOADING -----------------
@st.cache_data
def load_data():
    df = pd.read_csv("software_developers_onet_experience.csv")
    dfc = pd.read_csv("software_developers_onet_coefficients.csv")
    return df, dfc

df, dfc = load_data()

cog_cols = [c for c in dfc.columns if c not in ["Task", "Formula"]]
task_cols = [c for c in df.columns if c.startswith("Exp_")]
tasks_dict = dict(zip(task_cols, dfc["Task"]))

# ----------- MOCK PROJECTS STRUCTURE -----------
project_names = ["Website Redesign", "AI Assistant", "Mobile Platform"]
if "project_tasks" not in st.session_state:
    # Each project: dict of {task_col: assigned employee index}
    st.session_state.project_tasks = {
        "Website Redesign": {t: np.random.randint(0, len(df)) for t in task_cols},
        "AI Assistant": {t: np.random.randint(0, len(df)) for t in task_cols},
        "Mobile Platform": {t: np.random.randint(0, len(df)) for t in task_cols},
    }

# ---------- UTILS -------------------
def get_est_perf(row, task, coeffs):
    vals = np.array([row[a] for a in cog_cols])
    coefs = np.array([coeffs[a] for a in cog_cols])
    return np.dot(vals, coefs) * row[task]

def get_potential_perf(row, coeffs):
    vals = np.array([row[a] for a in cog_cols])
    coefs = np.array([coeffs[a] for a in cog_cols])
    return np.dot(vals, coefs) * 1.0

# --------- STYLES --------------
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; }
    .big-header { font-size:2.3em; font-weight:700; margin-bottom: 0.1em;}
    .menu-bar {background: #fff; border-bottom: 1px solid #eee; margin-bottom: 1em;}
    .menu-bar a {margin-right:30px; text-decoration:none; color: #444; font-weight:500;}
    .icon {font-size:1.6em; margin-right:6px;}
    .card {background: #fafbfc; border-radius: 14px; padding: 22px 28px; margin-bottom: 14px; box-shadow:0 2px 10px #eee;}
    .kpi {font-size: 2.1em; font-weight:700; color: #0070c0;}
    .act-feed .emoji {font-size: 1.2em;}
    .smallcaps {font-variant: small-caps;}
    </style>
""", unsafe_allow_html=True)

# --------- MENU ----------------
menu_opt = st.sidebar.radio("", ["🏠 Home", "👥 Team", "📂 Projects"])

# -------- HOME PAGE -------------
if menu_opt == "🏠 Home":
    st.markdown('<div class="menu-bar"><span class="icon">🏠</span>Home <span class="icon">👥</span>Team <span class="icon">📂</span>Projects <span class="icon">⚙️</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="big-header">Good morning. <span style="font-size:0.6em;color:#888;">{pd.Timestamp.now():%A, %B %d}</span></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.2, 1, 1])
    # -- Recent Activity --
    with col1:
        st.markdown('<div class="card act-feed"><b>Recent Activity</b><br>'
            '✅ Maria assigned to "System Design"<br>'
            '🐞 Ryan fixed "Testing/Debugging"<br>'
            '🔐 Security Review overdue for Alex<br>'
            '🔄 Kevin updated project "Website"<br>'
            '🆕 Jennifer joined<br>'
            '...</div>', unsafe_allow_html=True)
    # -- Workload Balance + Capacity --
    with col2:
        # Workload balance: stddev/mean gauge
        tasks_per = pd.Series([sum([st.session_state.project_tasks[proj][t]==i for proj in project_names for t in task_cols]) for i in range(len(df))])
        imbalance = tasks_per.std() / (tasks_per.mean() + 0.01)
        st.markdown(f'<div class="card"><b>Workload Balance</b><br><span class="kpi">{imbalance:.2f}</span> <br><span style="color:#888;">(Lower is more balanced)</span></div>', unsafe_allow_html=True)
        # Workload capacity: percent of recommended max
        max_per_person = 8
        total_tasks = len(project_names)*len(task_cols)
        assigned = len(project_names)*len(task_cols)
        usage = assigned / (len(df)*max_per_person)
        st.markdown(f'<div class="card"><b>Workload Capacity</b><br>'
                    f'<span class="kpi">{int(usage*100)}%</span> <br><span style="color:#888;">({assigned} tasks / {len(df)*max_per_person} capacity)</span></div>', unsafe_allow_html=True)
    # -- Suggestions --
    with col3:
        st.markdown('<div class="card"><b>Suggestions</b><br>'
            '🎯 <b>Training</b>: Mike could benefit from Project Training.<br>'
            '📈 <b>Quick Insights</b>: KPI = 85% <span style="color:green;">(+7.5% vs last week)</span><br>'
            '⚠️ <b>Overdue</b>: Security Review. <a href="#">Review now</a></div>', unsafe_allow_html=True)
    # -- Team Performance Chart --
    st.markdown('<div class="card" style="margin-top:0.7em;"><b>Team Performance</b> (monthly)</div>', unsafe_allow_html=True)
    months = ["Jan", "Feb", "Mar", "Apr", "May"]
    perf = np.clip(np.cumsum(np.random.randn(5)*2 + 73), 65, 97)
    fig, ax = plt.subplots(figsize=(5,2))
    ax.plot(months, perf, marker="o", lw=2)
    ax.set_ylim(60,100)
    ax.set_ylabel("Performance")
    ax.set_xlabel("Month")
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=0.18)
    st.pyplot(fig)

# ------------- TEAM PAGE -------------
elif menu_opt == "👥 Team":
    st.markdown('<div class="menu-bar"><span class="icon">🏠</span>Home <span class="icon">👥</span>Team <span class="icon">📂</span>Projects <span class="icon">⚙️</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="big-header">Team</div>', unsafe_allow_html=True)
    st.dataframe(df[["Name", "Surname", "Job_Position", "Years_Experience", "Project_Management_Experience", "Team_Leadership_Experience"]], use_container_width=True)

# ------------ PROJECTS PAGE -------------
elif menu_opt == "📂 Projects":
    st.markdown('<div class="menu-bar"><span class="icon">🏠</span>Home <span class="icon">👥</span>Team <span class="icon">📂</span>Projects <span class="icon">⚙️</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="big-header">Projects</div>', unsafe_allow_html=True)
    # Choose project
    proj = st.selectbox("Choose a project:", project_names)
    st.markdown(f"<b>Project:</b> {proj}", unsafe_allow_html=True)
    # Show current assignments table
    data = []
    for task in task_cols:
        assignee_idx = st.session_state.project_tasks[proj][task]
        coeffs = dfc[dfc["Task"]==tasks_dict[task]].iloc[0]
        est = get_est_perf(df.iloc[assignee_idx], task, coeffs)
        data.append([tasks_dict[task], f"{df.iloc[assignee_idx]['Name']} {df.iloc[assignee_idx]['Surname']}", est])
    assign_df = pd.DataFrame(data, columns=["Task", "Assigned To", "Est. Performance"])
    st.dataframe(assign_df, use_container_width=True)
    # --- Reassign Task ---
    st.markdown("#### Reassign a Task")
    tsel = st.selectbox("Pick a task to reassign:", task_cols, format_func=lambda x: tasks_dict[x])
    coeffs = dfc[dfc["Task"]==tasks_dict[tsel]].iloc[0]
    df["est_perf"] = df.apply(lambda r: get_est_perf(r, tsel, coeffs), axis=1)
    df["potential_perf"] = df.apply(lambda r: get_potential_perf(r, coeffs), axis=1)
    st.markdown("Top 5 Actual:")
    st.table(df.sort_values("est_perf", ascending=False)[["Name","Surname","Job_Position","est_perf"]].head(5))
    st.markdown("Top 5 Potential (if experience=1):")
    st.table(df.sort_values("potential_perf", ascending=False)[["Name","Surname","Job_Position","potential_perf"]].head(5))
    new_person = st.selectbox("Assign to:", df["Name"] + " " + df["Surname"])
    if st.button("Reassign"):
        idx = df.index[df["Name"] + " " + df["Surname"] == new_person][0]
        st.session_state.project_tasks[proj][tsel] = idx
        st.success("Reassigned!")

# END OF APP
