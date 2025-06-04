import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Sample Data (from before) ---
employee_data = [
    [1,"Alex Johnson","Backend Developer",0.8,0.7,0.5,0.6,0.4,0.9,0.7,0.8,0.8,0.7],
    [2,"Emily Clark","Frontend Developer",0.7,0.6,0.8,0.7,0.3,0.8,0.8,0.7,0.9,0.9],
    [3,"George Smith","Backend Developer",0.5,0.9,0.3,0.5,0.6,0.9,0.5,0.9,0.7,0.6],
    [4,"Christina Lee","Frontend Developer",0.6,0.8,0.7,0.8,0.2,0.7,0.7,0.8,0.8,0.8],
    [5,"Paul Brown","Data Manager",0.9,0.8,0.4,0.7,0.5,0.9,0.6,0.9,0.9,0.7],
    [6,"Nicholas Miller","QA Engineer",0.4,0.7,0.3,0.9,0.6,0.7,0.5,0.9,0.8,0.6],
    [7,"Marina Evans","Product Owner",0.8,0.8,0.9,0.8,0.3,0.7,0.9,0.8,0.9,0.9],
    [8,"Thomas Williams","Backend Developer",0.6,0.7,0.4,0.5,0.4,0.8,0.6,0.8,0.7,0.6],
    [9,"Sophia Martin","Frontend Developer",0.7,0.9,0.6,0.6,0.2,0.8,0.8,0.9,0.8,0.8],
    [10,"Elias Harris","QA Engineer",0.5,0.8,0.5,0.7,0.5,0.7,0.6,0.8,0.8,0.7],
    [11,"David Thompson","Backend Developer",0.6,0.8,0.4,0.6,0.3,0.8,0.7,0.8,0.8,0.8],
    [12,"Ashley Wright","Frontend Developer",0.8,0.7,0.7,0.7,0.4,0.8,0.8,0.8,0.9,0.8],
    [13,"Brian King","DevOps Engineer",0.7,0.9,0.5,0.6,0.5,0.9,0.6,0.9,0.7,0.8],
    [14,"Laura Scott","Data Analyst",0.9,0.8,0.6,0.8,0.3,0.9,0.7,0.9,0.8,0.9],
    [15,"Daniel Green","Frontend Developer",0.7,0.6,0.7,0.7,0.4,0.8,0.8,0.7,0.8,0.7],
    [16,"Rebecca Young","Backend Developer",0.8,0.9,0.5,0.6,0.4,0.9,0.7,0.9,0.8,0.8],
    [17,"James Hall","QA Engineer",0.5,0.7,0.4,0.8,0.5,0.8,0.5,0.9,0.8,0.7],
    [18,"Karen Moore","Product Owner",0.9,0.8,0.8,0.7,0.2,0.8,0.9,0.9,0.8,0.9],
    [19,"Michael Carter","Backend Developer",0.6,0.8,0.5,0.7,0.3,0.8,0.7,0.8,0.7,0.8],
    [20,"Jessica Turner","QA Engineer",0.7,0.7,0.6,0.8,0.4,0.8,0.7,0.8,0.8,0.8],
]
columns = ["ID","Name","Role","Openness","Conscientiousness","Extraversion","Agreeableness","Neuroticism",
           "Analytical_Thinking","Creativity","Attention_to_Detail","Learning_Agility","Verbal_Reasoning"]
df = pd.DataFrame(employee_data, columns=columns)

# --- Ideal profiles per role ---
ideal_profiles = {
    "Backend Developer":     {"Openness":0.7,"Conscientiousness":0.8,"Extraversion":0.4,"Agreeableness":0.6,"Neuroticism":0.4,"Analytical_Thinking":0.9,"Creativity":0.7,"Attention_to_Detail":0.8,"Learning_Agility":0.8,"Verbal_Reasoning":0.7},
    "Frontend Developer":    {"Openness":0.8,"Conscientiousness":0.7,"Extraversion":0.7,"Agreeableness":0.7,"Neuroticism":0.4,"Analytical_Thinking":0.7,"Creativity":0.9,"Attention_to_Detail":0.7,"Learning_Agility":0.8,"Verbal_Reasoning":0.8},
    "Data Manager":          {"Openness":0.8,"Conscientiousness":0.9,"Extraversion":0.5,"Agreeableness":0.7,"Neuroticism":0.3,"Analytical_Thinking":0.9,"Creativity":0.7,"Attention_to_Detail":0.9,"Learning_Agility":0.8,"Verbal_Reasoning":0.7},
    "QA Engineer":           {"Openness":0.6,"Conscientiousness":0.9,"Extraversion":0.4,"Agreeableness":0.8,"Neuroticism":0.3,"Analytical_Thinking":0.8,"Creativity":0.6,"Attention_to_Detail":0.9,"Learning_Agility":0.7,"Verbal_Reasoning":0.6},
    "Product Owner":         {"Openness":0.8,"Conscientiousness":0.8,"Extraversion":0.8,"Agreeableness":0.8,"Neuroticism":0.3,"Analytical_Thinking":0.8,"Creativity":0.8,"Attention_to_Detail":0.8,"Learning_Agility":0.9,"Verbal_Reasoning":0.9},
    "DevOps Engineer":       {"Openness":0.7,"Conscientiousness":0.9,"Extraversion":0.5,"Agreeableness":0.7,"Neuroticism":0.4,"Analytical_Thinking":0.9,"Creativity":0.7,"Attention_to_Detail":0.9,"Learning_Agility":0.8,"Verbal_Reasoning":0.7},
    "Data Analyst":          {"Openness":0.8,"Conscientiousness":0.9,"Extraversion":0.5,"Agreeableness":0.7,"Neuroticism":0.3,"Analytical_Thinking":0.9,"Creativity":0.7,"Attention_to_Detail":0.9,"Learning_Agility":0.8,"Verbal_Reasoning":0.8}
}

attribute_labels = ["Openness","Conscientiousness","Extraversion","Agreeableness","Neuroticism",
                   "Analytical_Thinking","Creativity","Attention_to_Detail","Learning_Agility","Verbal_Reasoning"]

# Dummy performance for main dashboard
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
performance = {
    'Current':  [70, 68, 75, 72, 77, 80, 78, 79, 82, 84, 85, 85],
    'Predicted':[72, 73, 76, 78, 81, 82, 82, 84, 86, 88, 88, 88],
    'Ideal':    [80, 80, 80, 80, 80, 85, 85, 85, 90, 90, 90, 90]
}
performance_df = pd.DataFrame(performance, index=months)

# --- Session State Setup ---
if "page" not in st.session_state:
    st.session_state.page = "dashboard"
if "selected_employee_id" not in st.session_state:
    st.session_state.selected_employee_id = None

def go_profiles():
    st.session_state.page = "profiles"

def go_dashboard():
    st.session_state.page = "dashboard"
    st.session_state.selected_employee_id = None

def go_employee(emp_id):
    st.session_state.selected_employee_id = emp_id
    st.session_state.page = "employee"

# --- Central Dashboard ---
if st.session_state.page == "dashboard":
    st.header("Dashboard - Middle Manager View")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Performance Over Time")
        fig, ax = plt.subplots()
        ax.plot(months, performance_df['Current'], label='Current', color='black', marker='o')
        ax.plot(months, performance_df['Predicted'], label='Predicted', color='red', linestyle='--', marker='o')
        ax.plot(months, performance_df['Ideal'], label='Ideal', color='gray', linestyle=':', marker='o')
        ax.set_xlabel("Month")
        ax.set_ylabel("Performance Score")
        ax.set_ylim(60, 100)
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with col2:
        st.subheader("Team Overall Fit (Radar)")
        radar_labels = attribute_labels
        mean_scores = [df[attr].mean() for attr in radar_labels]
        num_vars = len(radar_labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        mean_scores += mean_scores[:1]
        angles += angles[:1]
        fig2, ax2 = plt.subplots(subplot_kw=dict(polar=True))
        ax2.plot(angles, mean_scores, color='blue', linewidth=2)
        ax2.fill(angles, mean_scores, color='skyblue', alpha=0.25)
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(radar_labels, fontsize=10)
        ax2.set_yticks([0.2,0.4,0.6,0.8,1.0])
        ax2.set_yticklabels(['0.2','0.4','0.6','0.8','1.0'], fontsize=8)
        ax2.set_title("Team Attribute Averages", y=1.1, fontsize=12)
        st.pyplot(fig2)

    st.markdown("---")
    nav1, nav2, nav3 = st.columns(3)
    with nav1:
        st.button("Employee Profiles", on_click=go_profiles)
    with nav2:
        st.button("Team Analytics")
    with nav3:
        st.button("Monthly Portal")

# --- Employee Profiles (Selection Grid) ---
elif st.session_state.page == "profiles":
    st.title("Employee Profiles")
    st.button("← Back to Dashboard", on_click=go_dashboard)
    # Show employees as clickable cards
    cols = st.columns(4)
    for idx, row in df.iterrows():
        with cols[idx%4]:
            st.button(f"{row['Name']}\n({row['Role']})", key=f"emp_{row['ID']}", on_click=go_employee, args=(row['ID'],))
    st.markdown("---")
    st.info("Select an employee to view their profile dashboard.")

# --- Employee Dashboard ---
elif st.session_state.page == "employee":
    emp = df[df["ID"]==st.session_state.selected_employee_id].iloc[0]
    st.button("← Back to Employee Profiles", on_click=go_profiles)
    st.header(f"{emp['Name']} - {emp['Role']}")
    st.write(f"**Department:** Example Department  \n**Tasks:** Example tasks, more can be added")
    layout = st.columns([1,1])
    top1, top2 = layout
    bottom1, bottom2 = st.columns([1,1])

    # --- Predicted performance gauge (dummy value) ---
    with top1:
        st.subheader("Predicted Performance")
        performance = np.random.randint(70, 96)
        fig, ax = plt.subplots(figsize=(3,1.5), subplot_kw={'aspect': 'equal'})
        theta = np.linspace(-np.pi/2, np.pi/2, 100)
        ax.plot(np.cos(theta), np.sin(theta), color="gray", linewidth=8)
        ax.plot([0, np.cos(-np.pi/2 + np.pi*performance/100)], [0, np.sin(-np.pi/2 + np.pi*performance/100)], color="red", linewidth=8)
        ax.text(0, -0.2, f"{performance}%", fontsize=22, ha='center')
        ax.set_axis_off()
        st.pyplot(fig)

    # --- Radar chart: employee vs ideal profile ---
    with top2:
        st.subheader("Role Fit (Radar)")
        role = emp['Role']
        ideal = [ideal_profiles.get(role, {}).get(attr,0.7) for attr in attribute_labels]
        actual = [emp[attr] for attr in attribute_labels]
        labels = attribute_labels
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        ideal += ideal[:1]; actual += actual[:1]; angles += angles[:1]
        fig2, ax2 = plt.subplots(subplot_kw=dict(polar=True))
        ax2.plot(angles, actual, label="Employee", color="blue", linewidth=2)
        ax2.fill(angles, actual, color="skyblue", alpha=0.3)
        ax2.plot(angles, ideal, label="Ideal", color="red", linewidth=2, linestyle="--")
        ax2.fill(angles, ideal, color="salmon", alpha=0.15)
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(labels, fontsize=9)
        ax2.set_yticks([0.2,0.4,0.6,0.8,1.0])
        ax2.set_yticklabels(['0.2','0.4','0.6','0.8','1.0'], fontsize=7)
        ax2.set_title("Fit to Role Profile", fontsize=12)
        ax2.legend(loc="upper right", bbox_to_anchor=(1.2,1.1))
        st.pyplot(fig2)
        fit = 1 - np.mean([abs(a - b) for a, b in zip(actual, ideal)])
        st.markdown(f"**Overall Fit:** `{fit*100:.1f}%`")

    # --- Psychological (Big 5) as sliders (left-bottom) ---
    with bottom1:
        st.subheader("Psychological Attributes")
        for attr in attribute_labels[:5]:
            st.progress(emp[attr], text=attr)

    # --- Cognitive bar chart (right-bottom) ---
    with bottom2:
        st.subheader("Cognitive Skills")
        cognitive = attribute_labels[5:]
        values = [emp[attr] for attr in cognitive]
        used = np.random.choice(len(cognitive), 2, replace=False)
        fig3, ax3 = plt.subplots()
        bars = ax3.bar(cognitive, values, color=["crimson" if i in used else "gray" for i in range(len(cognitive))])
        ax3.set_ylim(0,1)
        for idx in used:
            bars[idx].set_hatch('//')
        ax3.set_ylabel("Score")
        ax3.set_title("Cognitive Skills (most used highlighted)")
        plt.xticks(rotation=20)
        st.pyplot(fig3)

    st.caption("Demo dashboard. All data and 'currently most used' bars are random for now.")

