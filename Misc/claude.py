import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# Set page config
st.set_page_config(
    page_title="Human Capital Management Platform",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .task-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e6e6e6;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .employee-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    .workload-high { background-color: #ffebee; border-left-color: #f44336; }
    .workload-medium { background-color: #fff3e0; border-left-color: #ff9800; }
    .workload-low { background-color: #e8f5e8; border-left-color: #4caf50; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'employees' not in st.session_state:
    st.session_state.employees = {}
if 'projects' not in st.session_state:
    st.session_state.projects = []
if 'task_assignments' not in st.session_state:
    st.session_state.task_assignments = {}

# Task coefficients and formulas
TASK_COEFFS = {
    "Analyze information to determine, recommend, and plan installation of a new system or modification of an existing system.": {
        "Logical_Reasoning": 0.136, "Analytical_Thinking": 0.155, "Decision_Making": 0.117
    },
    "Analyze user needs and software requirements to determine feasibility of design within time and cost constraints.": {
        "Logical_Reasoning": 0.099, "Attention_to_Detail": 0.091, "Creativity": 0.099, 
        "Analytical_Thinking": 0.107, "Communication_Skills": 0.078, "Leadership_Potential": 0.099
    },
    "Confer with data processing or project managers to obtain information on limitations or capabilities for data processing projects.": {
        "Agreeableness": 0.053, "Abstract_Reasoning": 0.057, "Analytical_Thinking": 0.168, "Leadership_Potential": 0.135
    },
    "Confer with systems analysts, engineers, programmers and others to design systems and to obtain information on project limitations and capabilities, performance requirements and interfaces.": {
        "Processing_Speed": 0.117, "Attention_to_Detail": 0.111, "Creativity": 0.114, "Analytical_Thinking": 0.089
    },
    "Consult with customers or other departments on project status, proposals, or technical issues, such as software system design or maintenance.": {
        "Extraversion": 0.141, "Attention_to_Detail": 0.085, "Creativity": 0.11, "Communication_Skills": 0.122
    },
    "Coordinate installation of software system.": {
        "Cognitive_Flexibility": 0.064, "Working_Memory": 0.055, "Pattern_Recognition": 0.051, 
        "Communication_Skills": 0.191, "Leadership_Potential": 0.123
    },
    "Design, develop and modify software systems, using scientific analysis and mathematical models to predict and measure outcomes and consequences of design.": {
        "Cognitive_Flexibility": 0.103, "Working_Memory": 0.117, "Attention_to_Detail": 0.082, 
        "Problem_Solving": 0.076, "Creativity": 0.114
    },
    "Determine system performance standards.": {
        "Processing_Speed": 0.168, "Creativity": 0.053, "Analytical_Thinking": 0.099
    },
    "Develop or direct software system testing or validation procedures, programming, or documentation.": {
        "Conscientiousness": 0.054, "Emotional_Stability": 0.054, "Working_Memory": 0.129, 
        "Attention_to_Detail": 0.147, "Problem_Solving": 0.094
    },
    "Modify existing software to correct errors, adapt it to new hardware, or upgrade interfaces and improve performance.": {
        "Cognitive_Flexibility": 0.144, "Processing_Speed": 0.091, "Problem_Solving": 0.101, "Analytical_Thinking": 0.097
    },
    "Monitor functioning of equipment to ensure system operates in conformance with specifications.": {
        "Neuroticism": 0.057, "Attention_to_Detail": 0.122, "Problem_Solving": 0.057, "Creativity": 0.053, 
        "Stress_Tolerance": 0.131, "Decision_Making": 0.053, "Leadership_Potential": 0.057
    },
    "Obtain and evaluate information on factors such as reporting formats required, costs, or security needs to determine hardware configuration.": {
        "Attention_to_Detail": 0.131, "Stress_Tolerance": 0.149, "Communication_Skills": 0.053
    },
    "Prepare reports or correspondence concerning project specifications, activities, or status.": {
        "Openness_to_Experience": 0.056, "Agreeableness": 0.052, "Emotional_Stability": 0.052, 
        "Cognitive_Flexibility": 0.056, "Analytical_Thinking": 0.121, "Stress_Tolerance": 0.06, "Communication_Skills": 0.161
    },
    "Recommend purchase of equipment to control dust, temperature, or humidity in area of system installation.": {
        "Conscientiousness": 0.052, "Extraversion": 0.057, "Agreeableness": 0.073, "Neuroticism": 0.057, 
        "Emotional_Stability": 0.073, "Processing_Speed": 0.067, "Attention_to_Detail": 0.057, 
        "Problem_Solving": 0.057, "Stress_Tolerance": 0.057
    },
    "Specify power supply requirements and configuration.": {
        "Extraversion": 0.059, "Cognitive_Flexibility": 0.051, "Working_Memory": 0.051, 
        "Attention_to_Detail": 0.102, "Analytical_Thinking": 0.157, "Decision_Making": 0.051
    },
    "Store, retrieve, and manipulate data for analysis of system capabilities and requirements.": {
        "Neuroticism": 0.054, "Attention_to_Detail": 0.144, "Analytical_Thinking": 0.129
    },
    "Supervise and assign work to programmers, designers, technologists, technicians, or other engineering or scientific personnel.": {
        "Attention_to_Detail": 0.086, "Creativity": 0.129, "Decision_Making": 0.104, "Leadership_Potential": 0.123
    },
    "Train users to use new or modified equipment.": {
        "Processing_Speed": 0.056, "Communication_Skills": 0.107, "Leadership_Potential": 0.159
    },
    "Modify code and hardware interfaces (if needed to reflect tasks above).": {
        "Cognitive_Flexibility": 0.137, "Pattern_Recognition": 0.053, "Problem_Solving": 0.115, "Communication_Skills": 0.053
    }
}

def load_sample_data():
    """Load sample employee data"""
    employees_data = {
        "Alex Johnson": {
            "position": "Software Engineer",
            "traits": {
                "Openness_to_Experience": 78, "Conscientiousness": 85, "Extraversion": 62, "Agreeableness": 71,
                "Neuroticism": 34, "Emotional_Stability": 76, "Cognitive_Flexibility": 82, "Working_Memory": 88,
                "Processing_Speed": 75, "Pattern_Recognition": 91, "Abstract_Reasoning": 79, "Logical_Reasoning": 86,
                "Attention_to_Detail": 94, "Problem_Solving": 83, "Creativity": 77, "Analytical_Thinking": 85,
                "Stress_Tolerance": 68, "Adaptability": 74, "Decision_Making": 81, "Communication_Skills": 59,
                "Leadership_Potential": 43
            },
            "experiences": [0.78, 0.9, 0.81, 0.97, 0.91, 0.92, 0.8, 0.9, 0.59, 0.56, 0.63, 0.8, 0.92, 0.61, 0.59, 0.57, 0.74, 0.74, 0.89],
            "assigned_tasks": []
        },
        "Maria Rodriguez": {
            "position": "Senior Software Engineer",
            "traits": {
                "Openness_to_Experience": 91, "Conscientiousness": 92, "Extraversion": 58, "Agreeableness": 83,
                "Neuroticism": 28, "Emotional_Stability": 84, "Cognitive_Flexibility": 89, "Working_Memory": 93,
                "Processing_Speed": 82, "Pattern_Recognition": 87, "Abstract_Reasoning": 94, "Logical_Reasoning": 91,
                "Attention_to_Detail": 97, "Problem_Solving": 89, "Creativity": 85, "Analytical_Thinking": 92,
                "Stress_Tolerance": 78, "Adaptability": 86, "Decision_Making": 88, "Communication_Skills": 67,
                "Leadership_Potential": 71
            },
            "experiences": [0.84, 0.92, 0.84, 0.94, 0.86, 0.89, 0.98, 0.98, 0.93, 0.89, 0.85, 0.9, 0.86, 0.82, 0.82, 0.91, 0.98, 0.95, 0.99],
            "assigned_tasks": []
        },
        "David Chen": {
            "position": "Frontend Developer",
            "traits": {
                "Openness_to_Experience": 68, "Conscientiousness": 74, "Extraversion": 76, "Agreeableness": 77,
                "Neuroticism": 42, "Emotional_Stability": 72, "Cognitive_Flexibility": 75, "Working_Memory": 79,
                "Processing_Speed": 88, "Pattern_Recognition": 73, "Abstract_Reasoning": 71, "Logical_Reasoning": 74,
                "Attention_to_Detail": 82, "Problem_Solving": 76, "Creativity": 91, "Analytical_Thinking": 78,
                "Stress_Tolerance": 65, "Adaptability": 79, "Decision_Making": 75, "Communication_Skills": 84,
                "Leadership_Potential": 55
            },
            "experiences": [0.7, 0.48, 0.89, 0.65, 0.56, 0.64, 0.79, 0.65, 0.58, 0.81, 0.5, 0.7, 0.78, 0.57, 0.61, 0.67, 0.54, 0.9, 0.85],
            "assigned_tasks": []
        },
        "Sarah Williams": {
            "position": "Backend Developer",
            "traits": {
                "Openness_to_Experience": 82, "Conscientiousness": 89, "Extraversion": 45, "Agreeableness": 68,
                "Neuroticism": 38, "Emotional_Stability": 79, "Cognitive_Flexibility": 86, "Working_Memory": 91,
                "Processing_Speed": 79, "Pattern_Recognition": 94, "Abstract_Reasoning": 88, "Logical_Reasoning": 92,
                "Attention_to_Detail": 89, "Problem_Solving": 91, "Creativity": 72, "Analytical_Thinking": 89,
                "Stress_Tolerance": 74, "Adaptability": 81, "Decision_Making": 84, "Communication_Skills": 52,
                "Leadership_Potential": 48
            },
            "experiences": [0.83, 0.73, 0.97, 0.74, 0.75, 0.84, 0.71, 0.96, 0.86, 0.96, 0.83, 0.92, 0.65, 0.86, 0.57, 0.97, 0.71, 0.6, 0.58],
            "assigned_tasks": []
        },
        "Kevin Moore": {
            "position": "Technical Lead",
            "traits": {
                "Openness_to_Experience": 86, "Conscientiousness": 91, "Extraversion": 74, "Agreeableness": 79,
                "Neuroticism": 33, "Emotional_Stability": 84, "Cognitive_Flexibility": 92, "Working_Memory": 94,
                "Processing_Speed": 87, "Pattern_Recognition": 93, "Abstract_Reasoning": 91, "Logical_Reasoning": 94,
                "Attention_to_Detail": 95, "Problem_Solving": 93, "Creativity": 81, "Analytical_Thinking": 95,
                "Stress_Tolerance": 85, "Adaptability": 92, "Decision_Making": 94, "Communication_Skills": 86,
                "Leadership_Potential": 89
            },
            "experiences": [0.89, 0.91, 0.94, 0.97, 0.82, 0.85, 0.82, 0.83, 0.98, 0.95, 0.82, 0.93, 0.98, 0.82, 0.84, 0.94, 0.98, 0.82, 0.84],
            "assigned_tasks": []
        }
    }
    return employees_data

def calculate_task_performance(employee_traits, task_coeffs, experience):
    """Calculate performance score for a task"""
    score = 0
    for trait, coeff in task_coeffs.items():
        if trait in employee_traits:
            score += coeff * employee_traits[trait]
    return score * experience

def get_top_performers(task_index, employees_data, num_performers=5):
    """Get top performers for a specific task"""
    task_name = list(TASK_COEFFS.keys())[task_index]
    task_coeffs = TASK_COEFFS[task_name]
    
    performances = []
    performances_exp1 = []
    
    for name, data in employees_data.items():
        # Actual performance with real experience
        actual_perf = calculate_task_performance(data['traits'], task_coeffs, data['experiences'][task_index])
        performances.append((name, actual_perf, data['experiences'][task_index]))
        
        # Performance if experience was 1.0
        exp1_perf = calculate_task_performance(data['traits'], task_coeffs, 1.0)
        performances_exp1.append((name, exp1_perf, 1.0))
    
    # Sort by performance score
    performances.sort(key=lambda x: x[1], reverse=True)
    performances_exp1.sort(key=lambda x: x[1], reverse=True)
    
    return performances[:num_performers], performances_exp1[:num_performers]

def calculate_workload_metrics(employees_data):
    """Calculate workload balance and capacity metrics"""
    task_counts = {}
    total_performance = {}
    
    for name, data in employees_data.items():
        task_count = len(data['assigned_tasks'])
        task_counts[name] = task_count
        
        # Calculate total performance across all assigned tasks
        total_perf = 0
        for task_idx in data['assigned_tasks']:
            task_name = list(TASK_COEFFS.keys())[task_idx]
            task_coeffs = TASK_COEFFS[task_name]
            perf = calculate_task_performance(data['traits'], task_coeffs, data['experiences'][task_idx])
            total_perf += perf
        total_performance[name] = total_perf
    
    # Calculate balance (standard deviation of task counts)
    counts = list(task_counts.values())
    if len(counts) > 1:
        balance_score = 100 - (np.std(counts) / (np.mean(counts) + 1) * 100)
        balance_score = max(0, min(100, balance_score))
    else:
        balance_score = 100
    
    # Calculate average capacity utilization
    avg_tasks = np.mean(counts) if counts else 0
    capacity_utilization = min(100, (avg_tasks / 5) * 100)  # Assuming 5 tasks is 100% capacity
    
    return {
        'task_counts': task_counts,
        'total_performance': total_performance,
        'balance_score': balance_score,
        'capacity_utilization': capacity_utilization,
        'avg_tasks_per_employee': avg_tasks
    }

# Sidebar navigation
st.sidebar.title("🏢 HCM Platform")
page = st.sidebar.selectbox("Navigate to:", ["🏠 Home", "👥 Team", "📁 Projects"])

# Load sample data if not already loaded
if not st.session_state.employees:
    st.session_state.employees = load_sample_data()

# HOME PAGE
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">Human Capital Management Dashboard</h1>', unsafe_allow_html=True)
    
    # Calculate workload metrics
    workload_metrics = calculate_workload_metrics(st.session_state.employees)
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Employees", len(st.session_state.employees))
    
    with col2:
        st.metric("Workload Balance", f"{workload_metrics['balance_score']:.1f}%")
    
    with col3:
        st.metric("Capacity Utilization", f"{workload_metrics['capacity_utilization']:.1f}%")
    
    with col4:
        st.metric("Avg Tasks/Employee", f"{workload_metrics['avg_tasks_per_employee']:.1f}")
    
    # Workload Distribution Chart
    st.subheader("📊 Workload Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Task count by employee
        task_counts_df = pd.DataFrame(list(workload_metrics['task_counts'].items()), 
                                    columns=['Employee', 'Task_Count'])
        
        fig_tasks = px.bar(task_counts_df, x='Employee', y='Task_Count', 
                          title='Number of Assigned Tasks per Employee',
                          color='Task_Count', color_continuous_scale='RdYlGn_r')
        fig_tasks.update_xaxis(tickangle=45)
        st.plotly_chart(fig_tasks, use_container_width=True)
    
    with col2:
        # Performance distribution
        perf_df = pd.DataFrame(list(workload_metrics['total_performance'].items()), 
                             columns=['Employee', 'Total_Performance'])
        
        fig_perf = px.bar(perf_df, x='Employee', y='Total_Performance', 
                         title='Total Performance Score by Employee',
                         color='Total_Performance', color_continuous_scale='Viridis')
        fig_perf.update_xaxis(tickangle=45)
        st.plotly_chart(fig_perf, use_container_width=True)
    
    # Employee Status Cards
    st.subheader("👤 Employee Status Overview")
    
    for name, data in st.session_state.employees.items():
        task_count = len(data['assigned_tasks'])
        
        # Determine workload level
        if task_count >= 4:
            workload_class = "workload-high"
            workload_status = "🔴 High"
        elif task_count >= 2:
            workload_class = "workload-medium"
            workload_status = "🟡 Medium"
        else:
            workload_class = "workload-low"
            workload_status = "🟢 Low"
        
        with st.container():
            st.markdown(f"""
            <div class="metric-card {workload_class}">
                <h4>{name} - {data['position']}</h4>
                <p><strong>Assigned Tasks:</strong> {task_count}</p>
                <p><strong>Workload Status:</strong> {workload_status}</p>
                <p><strong>Performance Score:</strong> {workload_metrics['total_performance'][name]:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

# TEAM PAGE
elif page == "👥 Team":
    st.markdown('<h1 class="main-header">Team Management</h1>', unsafe_allow_html=True)
    
    # Team overview
    st.subheader("👥 Team Overview")
    
    # Create team dataframe
    team_data = []
    for name, data in st.session_state.employees.items():
        team_data.append({
            'Name': name,
            'Position': data['position'],
            'Assigned Tasks': len(data['assigned_tasks']),
            'Avg Performance': np.mean([calculate_task_performance(data['traits'], 
                                      list(TASK_COEFFS.values())[i], 
                                      data['experiences'][i]) 
                                     for i in range(len(TASK_COEFFS))]),
            'Leadership Score': data['traits']['Leadership_Potential'],
            'Communication Score': data['traits']['Communication_Skills']
        })
    
    team_df = pd.DataFrame(team_data)
    st.dataframe(team_df, use_container_width=True)
    
    # Skills analysis
    st.subheader("🎯 Skills Analysis")
    
    # Skill distribution heatmap
    skills_data = []
    key_skills = ['Analytical_Thinking', 'Problem_Solving', 'Creativity', 'Leadership_Potential', 
                  'Communication_Skills', 'Attention_to_Detail']
    
    for name, data in st.session_state.employees.items():
        skill_row = {'Employee': name}
        for skill in key_skills:
            skill_row[skill] = data['traits'][skill]
        skills_data.append(skill_row)
    
    skills_df = pd.DataFrame(skills_data)
    skills_df = skills_df.set_index('Employee')
    
    fig_heatmap = px.imshow(skills_df.values, 
                           x=skills_df.columns, 
                           y=skills_df.index,
                           title='Team Skills Heatmap',
                           color_continuous_scale='RdYlGn',
                           aspect='auto')
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Individual employee details
    st.subheader("👤 Individual Employee Details")
    
    selected_employee = st.selectbox("Select Employee:", list(st.session_state.employees.keys()))
    
    if selected_employee:
        emp_data = st.session_state.employees[selected_employee]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Position:** {emp_data['position']}")
            st.write(f"**Assigned Tasks:** {len(emp_data['assigned_tasks'])}")
            
            # Top traits
            traits_sorted = sorted(emp_data['traits'].items(), key=lambda x: x[1], reverse=True)
            st.write("**Top 5 Traits:**")
            for trait, score in traits_sorted[:5]:
                st.write(f"- {trait}: {score}")
        
        with col2:
            # Performance radar chart
            traits_for_radar = ['Analytical_Thinking', 'Problem_Solving', 'Creativity', 
                              'Leadership_Potential', 'Communication_Skills', 'Attention_to_Detail']
            values = [emp_data['traits'][trait] for trait in traits_for_radar]
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=traits_for_radar,
                fill='toself',
                name=selected_employee
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=True,
                title=f"{selected_employee} - Skills Profile"
            )
            st.plotly_chart(fig_radar, use_container_width=True)

# PROJECTS PAGE
elif page == "📁 Projects":
    st.markdown('<h1 class="main-header">Project Management & Task Assignment</h1>', unsafe_allow_html=True)
    
    # Task Assignment Section
    st.subheader("🎯 Assign Tasks to Team Members")
    
    # Task selection
    task_names = list(TASK_COEFFS.keys())
    selected_task_idx = st.selectbox("Select Task:", 
                                   range(len(task_names)), 
                                   format_func=lambda x: f"Task {x+1}: {task_names[x][:60]}...")
    
    selected_task_name = task_names[selected_task_idx]
    st.write(f"**Selected Task:** {selected_task_name}")
    
    # Get top performers
    top_actual, top_potential = get_top_performers(selected_task_idx, st.session_state.employees)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top 5 Performers (Current Experience)")
        for i, (name, score, exp) in enumerate(top_actual):
            st.markdown(f"""
            <div class="employee-card">
                <h4>{i+1}. {name}</h4>
                <p><strong>Performance Score:</strong> {score:.2f}</p>
                <p><strong>Experience Level:</strong> {exp:.2f}</p>
                <p><strong>Position:</strong> {st.session_state.employees[name]['position']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("💎 Top 5 Potential (If Experience = 1.0)")
        for i, (name, score, exp) in enumerate(top_potential):
            st.markdown(f"""
            <div class="employee-card">
                <h4>{i+1}. {name}</h4>
                <p><strong>Potential Score:</strong> {score:.2f}</p>
                <p><strong>Current Experience:</strong> {st.session_state.employees[name]['experiences'][selected_task_idx]:.2f}</p>
                <p><strong>Position:</strong> {st.session_state.employees[name]['position']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Assignment interface
    st.subheader("📝 Make Assignment")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Combine all candidates
        all_candidates = set([name for name, _, _ in top_actual] + [name for name, _, _ in top_potential])
        selected_assignee = st.selectbox("Select Employee to Assign:", list(all_candidates))
    
    with col2:
        if st.button("Assign Task", type="primary"):
            if selected_assignee and selected_task_idx not in st.session_state.employees[selected_assignee]['assigned_tasks']:
                st.session_state.employees[selected_assignee]['assigned_tasks'].append(selected_task_idx)
                st.success(f"Task assigned to {selected_assignee}!")
                st.rerun()
            elif selected_task_idx in st.session_state.employees[selected_assignee]['assigned_tasks']:
                st.warning("This task is already assigned to this employee!")
    
    # Current assignments
    st.subheader("📋 Current Task Assignments")
    
    assignments_data = []
    for name, data in st.session_state.employees.items():
        for task_idx in data['assigned_tasks']:
            task_name = task_names[task_idx]
            task_coeffs = TASK_COEFFS[task_name]
            performance = calculate_task_performance(data['traits'], task_coeffs, data['experiences'][task_idx])
            
            assignments_data.append({
                'Employee': name,
                'Position': data['position'],
                'Task': f"Task {task_idx+1}",
                'Task Description': task_name[:50] + "...",
                'Performance Score': round(performance, 2),
                'Experience Level': data['experiences'][task_idx]
            })
    
    if assignments_data:
        assignments_df = pd.DataFrame(assignments_data)
        st.dataframe(assignments_df, use_container_width=True)
        
        # Remove assignment functionality
        st.subheader("🗑️ Remove Assignment")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if assignments_data:
                assignment_to_remove = st.selectbox(
                    "Select assignment to remove:",
                    range(len(assignments_data)),
                    format_func=lambda x: f"{assignments_data[x]['Employee']} - {assignments_data[x]['Task']}"
                )
        
        with col2:
            if st.button("Remove Assignment", type="secondary"):
                selected_assignment = assignments_data[assignment_to_remove]
                employee_name = selected_assignment['Employee']
                task_idx = int(selected_assignment['Task'].split(' ')[1]) -