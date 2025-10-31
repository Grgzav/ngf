import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ---- Company logo ----
logo = Image.open('microsoft.png')
st.image(logo, width=160)

# ---- Attribute definitions ----
attributes = ['Analytical', 'Detail', 'Verbal', 'Learning', 'Emotional', 'Adapt', 'Stress']
big5 = ['Big5_Openness', 'Big5_Conscientiousness', 'Big5_Extraversion', 'Big5_Agreeableness', 'Big5_Neuroticism']

departments = [
    "Engineering", "IT", "Operations", "Product Management", "HR (Human Resources)", "Marketing", "Sales",
    "Customer Support", "Quality Assurance", "Administration", "Compliance/Legal", "Finance", "Supply Chain",
    "Technical Support", "Learning & Development"
]
all_tasks = [
    "Problem Solving", "Maintenance", "Technical Troubleshooting", "System Monitoring",
    "Process Optimization", "Product Testing", "Workflow Automation",
    "Reporting & Documentation", "Inventory Management", "Project Planning", "Resource Allocation",
    "Quality Assurance", "Event Coordination", "Market Research", "Content Creation", "Sales Outreach",
    "Customer Feedback Collection", "Training & Onboarding", "Recruitment", "Compliance Review",
    "Data Entry", "Risk Assessment", "Budget Planning", "Client Support"
]
roles_tasks = {
    "Engineering": ["Problem Solving", "Maintenance", "Technical Troubleshooting", "System Monitoring",
                    "Process Optimization", "Product Testing", "Workflow Automation"],
    "IT": ["System Monitoring", "Technical Troubleshooting", "Maintenance", "Workflow Automation", "Reporting & Documentation"],
    "Operations": ["Maintenance", "Inventory Management", "Process Optimization", "Project Planning",
                   "System Monitoring", "Resource Allocation", "Quality Assurance", "Event Coordination"],
    "Product Management": ["Problem Solving", "Project Planning", "Process Optimization", "Market Research",
                           "Product Testing", "Content Creation", "Reporting & Documentation", "Risk Assessment", "Resource Allocation"],
    "HR (Human Resources)": ["Training & Onboarding", "Recruitment", "Compliance Review", "Data Entry",
                             "Reporting & Documentation", "Project Planning", "Resource Allocation", "Event Coordination"],
    "Marketing": ["Market Research", "Content Creation", "Sales Outreach", "Customer Feedback Collection", "Event Coordination"],
    "Sales": ["Sales Outreach", "Client Support", "Market Research", "Customer Feedback Collection"],
    "Customer Support": ["Client Support", "Technical Troubleshooting", "Customer Feedback Collection"],
    "Quality Assurance": ["Quality Assurance", "Product Testing", "Process Optimization", "Reporting & Documentation"],
    "Administration": ["Data Entry", "Inventory Management", "Reporting & Documentation", "Event Coordination", "Compliance Review", "Budget Planning"],
    "Compliance/Legal": ["Compliance Review", "Risk Assessment", "Reporting & Documentation"],
    "Finance": ["Data Entry", "Budget Planning", "Reporting & Documentation", "Risk Assessment"],
    "Supply Chain": ["Inventory Management", "Resource Allocation", "Reporting & Documentation"],
    "Technical Support": ["Technical Troubleshooting", "Client Support", "System Monitoring"],
    "Learning & Development": ["Training & Onboarding", "Content Creation", "Event Coordination"],
}
task_weights = {
    'Problem Solving':            [0.35, 0.10, 0.10, 0.15, 0.05, 0.15, 0.10],
    'Maintenance':                [0.10, 0.35, 0.05, 0.10, 0.05, 0.10, 0.25],
    'Technical Troubleshooting':  [0.25, 0.20, 0.10, 0.15, 0.05, 0.10, 0.15],
    'System Monitoring':          [0.10, 0.40, 0.05, 0.10, 0.05, 0.10, 0.20],
    'Process Optimization':       [0.30, 0.15, 0.10, 0.15, 0.05, 0.15, 0.10],
    'Product Testing':            [0.15, 0.30, 0.10, 0.10, 0.05, 0.10, 0.20],
    'Workflow Automation':        [0.30, 0.20, 0.10, 0.15, 0.00, 0.15, 0.10],
    'Reporting & Documentation':  [0.10, 0.30, 0.15, 0.10, 0.05, 0.10, 0.20],
    'Inventory Management':       [0.10, 0.30, 0.05, 0.10, 0.05, 0.10, 0.30],
    'Project Planning':           [0.20, 0.15, 0.10, 0.15, 0.10, 0.20, 0.10],
    'Resource Allocation':        [0.20, 0.15, 0.05, 0.10, 0.05, 0.15, 0.30],
    'Quality Assurance':          [0.15, 0.35, 0.05, 0.10, 0.05, 0.10, 0.20],
    'Event Coordination':         [0.05, 0.10, 0.10, 0.10, 0.25, 0.30, 0.10],
    'Market Research':            [0.30, 0.10, 0.20, 0.20, 0.05, 0.10, 0.05],
    'Content Creation':           [0.10, 0.10, 0.35, 0.10, 0.10, 0.15, 0.10],
    'Sales Outreach':             [0.05, 0.05, 0.40, 0.10, 0.20, 0.10, 0.10],
    'Customer Feedback Collection':[0.05, 0.05, 0.25, 0.05, 0.30, 0.20, 0.10],
    'Training & Onboarding':      [0.05, 0.10, 0.35, 0.10, 0.25, 0.10, 0.05],
    'Recruitment':                [0.05, 0.05, 0.35, 0.05, 0.35, 0.10, 0.05],
    'Compliance Review':          [0.15, 0.35, 0.05, 0.05, 0.05, 0.10, 0.25],
    'Data Entry':                 [0.05, 0.60, 0.05, 0.05, 0.05, 0.10, 0.10],
    'Risk Assessment':            [0.35, 0.20, 0.05, 0.10, 0.05, 0.10, 0.15],
    'Budget Planning':            [0.25, 0.25, 0.10, 0.10, 0.05, 0.10, 0.15],
    'Client Support':             [0.05, 0.05, 0.25, 0.05, 0.30, 0.20, 0.10],
}

# 30 Employees (auto-generated, 200 IQ assignments, age/big5 realistic)
np.random.seed(43)
names = [
    'Alice Smith','Brian Hall','Christine Green','Derek Lee','Ian Thomas','Emma Wilson','Oliver Evans','Sophie Carter',
    'Mason White','Julia White','Ethan Scott','Linda Adams','Mark King','Fiona Carter','Lucas Parker','Evelyn Turner',
    'Nathan Harris','Zoe Brooks','William Baker','Diana King','Aiden Wright','Megan Reed','Paul Simmons','Rachel Cooper',
    'Connor Price','Samantha Wood','Matthew Ross','Benjamin Bell','Grace Morris','Jack Russell'
]
department_assign = [
    "Engineering", "Engineering", "Engineering", "Engineering", "IT", "IT",
    "Operations", "Operations", "Operations", "Product Management", "Product Management",
    "HR (Human Resources)", "HR (Human Resources)", "Marketing", "Marketing", "Sales", "Sales",
    "Customer Support", "Customer Support", "Quality Assurance", "Quality Assurance", "Administration", "Administration",
    "Compliance/Legal", "Compliance/Legal", "Finance", "Finance", "Supply Chain", "Supply Chain", "Learning & Development"
]
ages = np.random.randint(28, 49, 30)
big5s = np.random.randint(5, 10, (30,5))
# Smart attribute distribution per department
def gen_attr(dept):
    if dept in ["Engineering","IT","Product Management"]:
        return [np.random.randint(7,10), np.random.randint(7,10), np.random.randint(6,9), np.random.randint(7,10),
                np.random.randint(5,8), np.random.randint(6,9), np.random.randint(5,8)]
    if dept in ["Marketing","Sales","HR (Human Resources)","Customer Support","Learning & Development"]:
        return [np.random.randint(6,9), np.random.randint(6,9), np.random.randint(8,11), np.random.randint(7,10),
                np.random.randint(7,10), np.random.randint(7,10), np.random.randint(6,9)]
    if dept in ["Quality Assurance","Administration","Compliance/Legal","Finance","Supply Chain"]:
        return [np.random.randint(7,10), np.random.randint(8,11), np.random.randint(6,8), np.random.randint(6,9),
                np.random.randint(5,8), np.random.randint(6,9), np.random.randint(7,10)]
    if dept in ["Operations","Technical Support"]:
        return [np.random.randint(7,10), np.random.randint(7,10), np.random.randint(6,8), np.random.randint(7,10),
                np.random.randint(6,9), np.random.randint(7,10), np.random.randint(7,10)]
    return [7,7,7,7,7,7,7]

employee_data = []
for i in range(30):
    name = names[i]
    dept = department_assign[i]
    attr = gen_attr(dept)
    age = ages[i]
    big5 = big5s[i].tolist()
    employee_data.append((name, dept, attr, age, big5))

data = []
for name, dept, attr, age, big5scores in employee_data:
    row = {'Employee': name, 'Department': dept, 'Age': age}
    for i, att in enumerate(attributes):
        row[att] = attr[i]
    for i, b5 in enumerate(big5):
        row[b5] = big5scores[i]
    data.append(row)
df_employees = pd.DataFrame(data)

# Predict task performance for all employees and all tasks
for task in all_tasks:
    df_employees[task] = (df_employees[attributes].values @ np.array(task_weights[task])) * 10
    df_employees[task] = df_employees[task].clip(upper=100)
for role, tasks in roles_tasks.items():
    df_employees[f'{role}_Fit%'] = df_employees[tasks].mean(axis=1)

# --- Sidebar navigation ---
tab = st.sidebar.radio("Navigation", ['Home', 'Team Analytics', 'Employee Analytics'])

if tab == 'Home':
    st.header("Team Performance Dashboard")
    st.markdown("#### Real vs Predicted Team Performance Over Time")

    # Simulate performance data
    np.random.seed(42)
    days = 60
    x = np.arange(days)
    # Real: team average, with bumps
    real = 60 + 8 * np.sin(x / 7) + 3 * np.random.randn(days) + 0.15 * x
    # Predicted: changes only when hiring/training
    predicted = 62 + 0.13 * x
    change_points = [0, 20, 40]
    step_vals = [62, 68, 74]
    for i, cp in enumerate(change_points[:-1]):
        predicted[cp:change_points[i+1]] = step_vals[i]
    predicted[change_points[-1]:] = step_vals[-1]

    # Plot in Streamlit
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(x, real, label="Real Performance", color="black", linewidth=2)
    ax.plot(x, predicted, label="Predicted Performance", color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("Day")
    ax.set_ylabel("Team Avg Performance (%)")
    ax.set_title("Team Real vs Predicted Performance Over Time")
    ax.legend()
    st.pyplot(fig)

    st.markdown("#### Current Department Performance")

    # Compute current department performance from actual employees
    dept_perf = {}
    for dept in departments:
        dept_emps = df_employees[df_employees['Department'] == dept]
        if len(dept_emps) > 0:
            dept_perf[dept] = dept_emps[[f"{dept}_Fit%"]].mean().values[0]
        else:
            dept_perf[dept] = 0.0

    # Show gauges in a grid
    ncols = 4
    rows = (len(departments) + ncols - 1) // ncols
    for row in range(rows):
        cols = st.columns(ncols)
        for i in range(ncols):
            idx = row*ncols + i
            if idx < len(departments):
                dept = departments[idx]
                perf = dept_perf[dept]
                with cols[i]:
                    fig, ax = plt.subplots(figsize=(2.2, 1.2), subplot_kw={'projection':'polar'})
                    theta_full = np.linspace(np.pi, 0, 100)
                    r = np.ones_like(theta_full)
                    ax.plot(theta_full, r, color='gray', lw=10, alpha=0.25, solid_capstyle='round')
                    value_theta = np.linspace(np.pi, np.pi - (perf/100)*np.pi, 100)
                    ax.plot(value_theta, np.ones_like(value_theta), color='tab:blue', lw=10, solid_capstyle='round')
                    t_needle = np.pi - (perf/100)*np.pi
                    ax.plot([t_needle, t_needle], [0, 1.1], color='red', lw=2, zorder=5)
                    ax.scatter(0,0, s=45, color='white', zorder=6, edgecolor='gray')
                    ax.text(0, -0.30, f"{perf:.1f}%", fontsize=11, fontweight='bold', ha='center', va='center')
                    ax.set_title(dept.replace(" (Human Resources)","").replace(" & ","\n"), fontsize=9, pad=7)
                    ax.set_yticklabels([])
                    ax.set_xticklabels([])
                    ax.set_theta_zero_location('E')
                    ax.set_theta_direction(1)
                    ax.set_rlim(0,1.15)
                    ax.axis('off')
                    st.pyplot(fig)

elif tab == 'Team Analytics':
    st.header("Team Analytics (Checkpoint 1 view)")

    col1, col2 = st.columns(2)
    with col1:
        selected_role = st.selectbox('Select Job Role', list(roles_tasks.keys()), key='role')
    with col2:
        selected_department = st.selectbox('Filter by Department', ['All'] + departments)
    selected_employees = st.multiselect(
        'Select up to 3 Employees',
        df_employees['Employee'] if selected_department == 'All'
        else df_employees[df_employees['Department'] == selected_department]['Employee'],
        default=None, max_selections=3, key='emp'
    )

    tasks_for_role = roles_tasks[selected_role]
    st.subheader(f'Job Role: {selected_role}')
    st.write('**Role Tasks:**', ', '.join(tasks_for_role))

    # Radar chart for selected employees (tasks = only role-relevant tasks)
    if len(selected_employees) > 0:
        angles = np.linspace(0, 2 * np.pi, len(tasks_for_role), endpoint=False).tolist()
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        colors = ['blue', 'green', 'purple']

        for idx, emp_name in enumerate(selected_employees):
            emp_row = df_employees[df_employees['Employee'] == emp_name].iloc[0]
            values = [emp_row[task] for task in tasks_for_role]
            values += values[:1]  # close the loop
            ax.plot(angles, values, color=colors[idx % len(colors)], label=emp_name, linewidth=2)
            ax.fill(angles, values, color=colors[idx % len(colors)], alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(tasks_for_role, rotation=15)
        ax.set_yticks(range(0, 101, 20))
        ax.set_ylim(0, 100)
        ax.set_title(f'Predicted Performance by Task: {selected_role}', size=16)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.12))
        st.pyplot(fig)
        st.subheader('Predicted Performance Scores (%)')
        display_cols = ['Employee', 'Department'] + tasks_for_role
        st.dataframe(df_employees[df_employees['Employee'].isin(selected_employees)][display_cols].reset_index(drop=True))
    else:
        st.info("Select at least one employee to view radar chart.")

    # Table: Best-fit employees for this role
    st.subheader(f"Best Fit Employees for '{selected_role}'")
    df_employees[f'{selected_role}_Fit%'] = df_employees[tasks_for_role].mean(axis=1)
    fit_table = df_employees[['Employee', 'Department', f'{selected_role}_Fit%'] + tasks_for_role].sort_values(f'{selected_role}_Fit%', ascending=False).reset_index(drop=True)
    fit_table[f'{selected_role}_Fit%'] = fit_table[f'{selected_role}_Fit%'].round(1)
    if selected_department != 'All':
        fit_table = fit_table[fit_table['Department'] == selected_department]
    st.dataframe(fit_table, use_container_width=True)

elif tab == 'Employee Analytics':
    st.header("Employee Analytics")
    selected_employee = st.selectbox('Select Employee', df_employees['Employee'])
    emp = df_employees[df_employees['Employee'] == selected_employee].iloc[0]
    dept = emp['Department']
    age = emp['Age']
    tasks = roles_tasks[dept]
    # Top rectangle
    st.markdown(
        f"""
        <div style="border-radius:12px;border:1px solid #DDD;padding:12px;background:#f5f5fa;width:100%">
        <h3 style="margin-bottom:0;">{emp['Employee']}</h3>
        <span style="font-weight:600;">Department:</span> {dept} <br>
        <span style="font-weight:600;">Age:</span> {age} <br>
        <span style="font-weight:600;">Relevant Tasks:</span> {', '.join(tasks)}
        </div>
        """, unsafe_allow_html=True
    )
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    # UL: Modern gauge (tachometer)
    with col1:
        avg_fit = emp[f"{dept}_Fit%"]
        fig, ax = plt.subplots(figsize=(4,2.5), subplot_kw={'projection':'polar'})

        theta_full = np.linspace(np.pi, 0, 100)
        r = np.ones_like(theta_full)
        ax.plot(theta_full, r, color='gray', lw=16, alpha=0.22, solid_capstyle='round')
        value_theta = np.linspace(np.pi, np.pi - (avg_fit/100)*np.pi, 100)
        ax.plot(value_theta, np.ones_like(value_theta), color='tab:blue', lw=16, solid_capstyle='round')
        t_needle = np.pi - (avg_fit/100)*np.pi
        ax.plot([t_needle, t_needle], [0, 1.12], color='red', lw=3, zorder=5)
        ax.scatter(0,0, s=140, color='white', zorder=6, edgecolor='gray')
        ax.text(0, -0.35, f"{avg_fit:.1f}%", fontsize=20, fontweight='bold', ha='center', va='center')
        ax.text(0, -0.7, "Job Role Fit", fontsize=12, ha='center', va='center', color='gray')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_theta_zero_location('E')
        ax.set_theta_direction(1)
        ax.set_rlim(0,1.2)
        ax.axis('off')
        st.pyplot(fig)
        # Value below gauge
        st.markdown(f"<h3 style='text-align:center; color:#1565c0; margin-top:-295px;margin-left:25px;'>{avg_fit:.1f}%</h3>", unsafe_allow_html=True)
    # UR: Spider for relevant tasks
    with col2:
        taskvals = [emp[t] for t in tasks]
        taskvals += taskvals[:1]
        angles = np.linspace(0, 2 * np.pi, len(tasks), endpoint=False).tolist()
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(polar=True))
        ax.plot(angles, taskvals, color='tab:blue', lw=2)
        ax.fill(angles, taskvals, color='tab:blue', alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(tasks, rotation=15)
        ax.set_yticks(range(0, 101, 20))
        ax.set_ylim(0, 100)
        ax.set_title("Relevant Task Performance")
        st.pyplot(fig)
    # LL: Bar chart cognitive attributes, highlight >0.3
    with col3:
        weights = np.array([max([task_weights[t][i] for t in tasks]) for i in range(len(attributes))])
        highlight = (weights > 0.3)
        colors = ['red' if h else 'tab:blue' for h in highlight]
        vals = [emp[a] for a in attributes]
        fig, ax = plt.subplots(figsize=(4,3))
        ax.bar(attributes, vals, color=colors)
        ax.set_ylim(0, 10)
        ax.set_title("Cognitive Attributes")
        st.pyplot(fig)
    # LR: Big 5 radar/bar chart
    with col4:
        b5vals = [emp[b] for b in big5]
        angles = np.linspace(0, 2 * np.pi, 5, endpoint=False).tolist()
        angles += angles[:1]
        b5vals += b5vals[:1]
        fig, ax = plt.subplots(figsize=(4.5,4.5), subplot_kw=dict(polar=True))
        ax.plot(angles, b5vals, color='tab:green', lw=2)
        ax.fill(angles, b5vals, color='tab:green', alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism'])
        ax.set_yticks(range(0, 11, 2))
        ax.set_ylim(0, 10)
        ax.set_title("Big 5 Personality Traits")
        st.pyplot(fig)
