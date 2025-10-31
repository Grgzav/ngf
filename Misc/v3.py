import streamlit as st
import pandas as pd
import altair as alt
import plotly.graph_objects as go
import streamlit.components.v1 as components

# --- Page config & CSS ---
st.set_page_config(page_title="Team Dashboard", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
body { background-color: #f4f4f4; font-family: 'Roboto', sans-serif; font-size: 16px; line-height: 1.5; }
.card { background: #ffffff; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); padding: 12px; margin-bottom: 12px; }
.badge { padding: 6px 10px; border-radius: 8px; font-size: 0.9rem; }
.badge--ontrack { background: #28a745; }
.badge--overdue { background: #dc3545; }
.section-title { font-size: 1.5rem; font-weight: 700; margin-bottom: 8px; }
.small-text { font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# --- Sample Data ---
projects = ["Authentication Module", "API Service", "UI Overhaul", "User Research"]
categories = ["Planning", "Development", "Testing", "Deployment", "Research"]
tasks = pd.DataFrame([
    {"Task Name":"Design Login Flow","Category":"Planning","Project Name":"Authentication Module","Assignee":"Alice Smith","Status":"On track","Due Date":"2025-07-25"},
    {"Task Name":"Implement API","Category":"Development","Project Name":"API Service","Assignee":"Bob Johnson","Status":"On track","Due Date":"2025-07-20"},
    {"Task Name":"Write Unit Tests","Category":"Testing","Project Name":"API Service","Assignee":"Carol Lee","Status":"Overdue","Due Date":"2025-07-27"},
    {"Task Name":"Prep Presentation","Category":"Deployment","Project Name":"UI Overhaul","Assignee":"David Kim","Status":"On track","Due Date":"2025-07-30"},
    {"Task Name":"Conduct UX Research","Category":"Research","Project Name":"User Research","Assignee":"Eva Wong","Status":"Overdue","Due Date":"2025-08-02"},
    {"Task Name":"Deploy to Staging","Category":"Deployment","Project Name":"Authentication Module","Assignee":"Alice Smith","Status":"On track","Due Date":"2025-07-28"},
    {"Task Name":"Optimize Database","Category":"Development","Project Name":"API Service","Assignee":"Carol Lee","Status":"On track","Due Date":"2025-08-05"},
    {"Task Name":"QA Regression Tests","Category":"Testing","Project Name":"UI Overhaul","Assignee":"David Kim","Status":"Overdue","Due Date":"2025-08-01"}
])

# Performance Data: 12 weeks
pred = [35,50,55,60,70,80,72,85,90,80,95,82]
actual = [sum(pred[i:i+4])/4 for i in range(0,len(pred),4) for _ in range(4)]
weeks = list(range(20, 32))
workload_vals = [50,65,40,75,55,60,45,70,50,65,55,80]
performance = pd.DataFrame({
    "Week": weeks,
    "Predicted %": pred,
    "Actual %": actual,
    "Workload %": workload_vals
})

# Metrics & Insights
hours_available = 52
hours_required = 65
fairness_score = 0.82
insights = [
    "Carol Lee is nearing capacity; consider reassigning tasks.",
    "Team outperformed predictions in week 29—momentum is strong!",
    "Schedule a UX workshop for Eva Wong to boost research speed."
]

# --- Layout ---
col1, col2 = st.columns(2, gap="small")
col3, col4 = st.columns(2, gap="small")

# Task Overview
with col1:
    st.markdown('<div class="card"><div class="section-title">Recent Task Activity</div></div>', unsafe_allow_html=True)
    header_html = (
        '<div style="display:grid;grid-template-columns:2fr 1fr 2fr 1fr 1fr 1fr;gap:8px;font-weight:600;margin-bottom:6px;">'
        '<div>Task Name</div><div>Category</div><div>Project Name</div><div>Assignee</div><div>Status</div><div>Due Date</div>'
        '</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)
    for _, row in tasks.iterrows():
        cls = 'badge--ontrack' if row['Status']=='On track' else 'badge--overdue'
        st.markdown(
            f'<div style="display:grid;grid-template-columns:2fr 1fr 2fr 1fr 1fr 1fr;gap:8px;align-items:center;margin-bottom:6px;">'
            f'<div>{row["Task Name"]}</div>'
            f'<div>{row["Category"]}</div>'
            f'<div>{row["Project Name"]}</div>'
            f'<div><small class="small-text">{row["Assignee"]}</small></div>'
            f'<div><span class="badge {cls}">{row["Status"]}</span></div>'
            f'<div><small class="small-text">{row["Due Date"]}</small></div>'
            '</div>', unsafe_allow_html=True
        )

# Performance Overview
if 'show_pred' not in st.session_state: st.session_state.show_pred=True
if 'show_work' not in st.session_state: st.session_state.show_work=False
with col2:
    st.markdown('<div class="card"><div class="section-title">Performance Overview</div></div>', unsafe_allow_html=True)
    x_axis=alt.Axis(title='Week',values=weeks,labelAngle=0,domain=False,ticks=True)
    y_left=alt.Axis(title='%',values=[0,25,50,75,100],grid=True,domain=False,ticks=True,format='.0f')
    y_right=alt.Axis(title='Workload',orient='right',domain=False,ticks=True)
    layers=[]
    if st.session_state.show_pred:
        layers.append(
            alt.Chart(performance).mark_bar(color='#4C78A8',size=20).encode(
                x=alt.X('Week:O',axis=x_axis),
                y=alt.Y('Predicted %:Q',axis=y_left)
            )
        )
    layers.append(
        alt.Chart(performance).mark_line(point=True,color='#F58518',size=3).encode(
            x=alt.X('Week:O',axis=x_axis),
            y=alt.Y('Actual %:Q',scale=alt.Scale(domain=[0,100]),axis=y_left)
        )
    )
    if st.session_state.show_work:
        layers.append(
            alt.Chart(performance).mark_line(point=True,color='black',strokeDash=[4,2],size=3).encode(
                x=alt.X('Week:O',axis=x_axis),
                y=alt.Y('Workload %:Q',scale=alt.Scale(domain=[0,100]),axis=y_right)
            )
        )
    combo = alt.layer(*layers).resolve_scale(y='independent').properties(height=250)
    combo = combo.configure_axis(labelFontSize=14, titleFontSize=16).configure_view(stroke=None)
    st.altair_chart(combo, use_container_width=True)
    btns = st.columns([1,2,2,1], gap="small")
    if btns[1].button("Toggle Predicted %"): st.session_state.show_pred = not st.session_state.show_pred
    if btns[2].button("Toggle Workload %"): st.session_state.show_work = not st.session_state.show_work

# Workload & Distribution
with col3:
    st.markdown('<div class="card"><div class="section-title">Workload & Distribution</div></div>', unsafe_allow_html=True)

    # a) Workload panel
    with st.container():
        avg_work = performance['Workload %'].mean()
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=avg_work,
            gauge={
                "axis": {"range":[0,100], "tickvals":[0,50,100]},
                "steps": [
                    {"range":[0,60],"color":"#4bbd5e"},
                    {"range":[60,80],"color":"#ffc107"},
                    {"range":[80,100],"color":"#dc3545"}
                ],
                "bar": {"color":"rgba(39,39,39,0.9)"},
                "threshold":{"line":{"color":"black","width":4},"thickness":1,"value":avg_work}
            }
        ))
        gauge.update_layout(
            margin=dict(t=10,b=10,l=0,r=0), height=280, paper_bgcolor="rgba(0,0,0,0)"
        )
        html = gauge.to_html(include_plotlyjs=False, full_html=False)
        frame_html = f"""
        <div style='margin:24px 0;padding:24px;border:1px solid #ddd;border-radius:12px;'>
          <h3 style='text-align:center;margin-bottom:16px;'>Workload</h3>
          {html}
        </div>
        """
        components.html(frame_html, height=350)

    # b) Workload Balance panel
    with st.container():
        pct = fairness_score * 100
        df_balance = pd.DataFrame([
            {'start': 0, 'end': 80, 'color': '#469ed1'},
            {'start': 80, 'end': 100, 'color': '#ddffdd'}
        ])
        chart = alt.Chart(df_balance).mark_bar(size=200).encode(
            y=alt.Y('start:Q', scale=alt.Scale(domain=[0,100]), title=None),
            y2='end:Q',
            x=alt.value(0),
            color=alt.Color('color:N', scale=None, legend=None)
        ).properties(width=120, height=280)
        rule = alt.Chart(pd.DataFrame({'value':[pct]})).mark_rule(color='black', strokeWidth=10).encode(
            y='value:Q', x=alt.value(0), x2=alt.value(120)
        )
        combined = (chart + rule).configure_view(stroke=None)
        alt_html = combined.to_html()
        frame_html2 = f"""
        <div style='margin:24px 0;padding:24px;border:1px solid #ddd;border-radius:12px;'>
          <h3 style='text-align:center;margin-bottom:16px;'>Workload Balance</h3>
          {alt_html}
        </div>
        """
        components.html(frame_html2, height=350)

    # Metrics below
    c1, c2 = st.columns(2, gap="small")
    with c1: st.metric("Hours Available", f"{hours_available}h")
    with c2: st.metric("Hours Required",  f"{hours_required}h")

# Insights
with col4:
    st.markdown('<div class="card"><div class="section-title">Insights & Suggestions</div></div>', unsafe_allow_html=True)
    for insight in insights:
        st.markdown(f"- {insight}")
