import streamlit as st
import requests
import json
import os
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, List
import time
import base64
from io import BytesIO
import numpy as np
import cv2
import tempfile

# Import our custom modules
from services.aimatchmaker import AIResumeEvaluator
from services.resumeparser import ResumeParser
from streamlit_extras.metric_cards import style_metric_cards

# Page config
st.set_page_config(
    page_title="SmartHire AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .skill-tag {
        display: inline-block;
        background: #667eea;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        margin: 0.1rem;
        font-size: 0.8rem;
    }
    
    .gap-tag {
        display: inline-block;
        background: #ff6b6b;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        margin: 0.1rem;
        font-size: 0.8rem;
    }
    
    .recommendation-box {
        background-color: #f8f9fa;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize our AI components
@st.cache_resource
def load_evaluator():
    return AIResumeEvaluator()

@st.cache_resource
def load_parser():
    return ResumeParser()

evaluator = load_evaluator()
parser = load_parser()

# Initialize session state
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None
if 'uploaded_resume_text' not in st.session_state:
    st.session_state.uploaded_resume_text = ""
if 'current_jd' not in st.session_state:
    st.session_state.current_jd = ""
if 'resume_info' not in st.session_state:
    st.session_state.resume_info = {}

def process_resume(uploaded_file):
    """Process uploaded resume file"""
    try:
        # Create a temporary file with proper extension
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            temp_path = tmp_file.name
        
        # Extract text from resume
        resume_text = parser.extract_text(temp_path)
        st.session_state.uploaded_resume_text = resume_text
        
        # Extract basic info
        resume_info = parser.extract_basic_info(resume_text)
        st.session_state.resume_info = resume_info
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        return resume_text, resume_info
    except Exception as e:
        st.error(f"Error processing resume: {str(e)}")
        return None, None

def evaluate_resume_with_ai(resume_text, jd_text):
    """Evaluate resume against job description using our AI model"""
    try:
        results = evaluator.evaluate(resume_text, jd_text)
        st.session_state.evaluation_results = results
        return results
    except Exception as e:
        st.error(f"Evaluation error: {str(e)}")
        return None

def create_radar_chart(results):
    """Create radar chart for skills visualization"""
    if not results:
        return None
        
    categories = ['Skill Match', 'Semantic Similarity', 'Experience Match']
    
    # Calculate experience match score
    exp_score = 0
    if results.get('required_experience', 0) > 0:
        exp_score = min(100, (results.get('experience_years', 0) / results.get('required_experience', 1)) * 100)
    
    values = [
        results.get('skill_match_score', 0),
        results.get('semantic_similarity', 0),
        exp_score
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Resume Analysis',
        line_color='#667eea'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        height=300,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    return fig

def create_skill_comparison_chart(resume_skills, jd_skills):
    """Create chart comparing skills"""
    if not resume_skills or not jd_skills:
        return None
        
    matched_skills = set(resume_skills) & set(jd_skills)
    resume_only = set(resume_skills) - set(jd_skills)
    jd_only = set(jd_skills) - set(resume_skills)
    
    data = {
        'Category': ['Matched Skills', 'Resume Only', 'JD Only'],
        'Count': [len(matched_skills), len(resume_only), len(jd_only)]
    }
    
    df = pd.DataFrame(data)
    
    fig = px.bar(df, x='Category', y='Count', 
                 color='Category',
                 color_discrete_map={
                     'Matched Skills': '#28a745',
                     'Resume Only': '#17a2b8',
                     'JD Only': '#ff6b6b'
                 })
    
    fig.update_layout(
        height=300,
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    return fig

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 SmartHire AI - Resume Evaluation System</h1>
        <p>AI-powered resume analysis and job matching</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Upload Resume")
        
        uploaded_file = st.file_uploader(
            "Choose a resume file",
            type=['pdf', 'docx', 'doc', 'txt', 'png', 'jpg', 'jpeg'],
            help="Supported formats: PDF, DOCX, DOC, TXT, PNG, JPG"
        )
        
        if uploaded_file is not None:
            with st.spinner("Processing resume..."):
                resume_text, resume_info = process_resume(uploaded_file)
                
            if resume_text:
                st.success("✅ Resume processed successfully!")
                
                # Display basic info
                if resume_info.get('email'):
                    st.write(f"**Email:** {resume_info['email']}")
                if resume_info.get('phone'):
                    st.write(f"**Phone:** {resume_info['phone']}")
                if resume_info.get('experience_years', 0) > 0:
                    st.write(f"**Experience:** {resume_info['experience_years']} years")
                if resume_info.get('skills'):
                    st.write(f"**Skills found:** {', '.join(resume_info['skills'][:5])}{'...' if len(resume_info['skills']) > 5 else ''}")
        
        st.header("📋 Job Description")
        jd_text = st.text_area(
            "Paste the job description here",
            height=200,
            value=st.session_state.current_jd,
            help="Copy and paste the complete job description for analysis"
        )
        
        st.session_state.current_jd = jd_text
        
        analyze_btn = st.button(
            "🚀 Analyze Resume vs Job Description",
            type="primary",
            disabled=not (uploaded_file and jd_text),
            use_container_width=True
        )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Resume Content")
        if st.session_state.uploaded_resume_text:
            with st.expander("View extracted resume text"):
                st.text(st.session_state.uploaded_resume_text[:1000] + "..." 
                        if len(st.session_state.uploaded_resume_text) > 1000 else st.session_state.uploaded_resume_text)
        else:
            st.info("Upload a resume to see the extracted content")
    
    with col2:
        st.subheader("Job Description")
        if st.session_state.current_jd:
            with st.expander("View job description"):
                st.text(st.session_state.current_jd[:1000] + "..." 
                        if len(st.session_state.current_jd) > 1000 else st.session_state.current_jd)
        else:
            st.info("Enter a job description to begin analysis")
    
    # Analyze button action
    if analyze_btn and uploaded_file and jd_text:
        with st.spinner("🤖 AI is analyzing your resume against the job description..."):
            results = evaluate_resume_with_ai(
                st.session_state.uploaded_resume_text, 
                st.session_state.current_jd
            )
            
            # Simulate processing time for better UX
            time.sleep(1)
    
    # Display results if available
    if st.session_state.evaluation_results and not st.session_state.evaluation_results.get('error'):
        results = st.session_state.evaluation_results
        
        st.markdown("---")
        st.header("📊 Evaluation Results")
        
        # Overall score
        score = results['overall_score']
        
        # Create columns for metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Overall Score</h3>
                <h2>{score}/100</h2>
                <p>{results['feedback']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Skill Match</h3>
                <h2>{results['skill_match_score']}%</h2>
                <p>{len(results['matched_skills'])}/{len(results['jd_skills'])} skills matched</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Semantic Similarity</h3>
                <h2>{results['semantic_similarity']}%</h2>
                <p>Content relevance to job description</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            exp_match = "✅ Meets requirement" if results['experience_years'] >= results['required_experience'] else "⚠️ Below requirement"
            st.markdown(f"""
            <div class="metric-card">
                <h3>Experience</h3>
                <h2>{results['experience_years']} yrs</h2>
                <p>{exp_match} ({results['required_experience']} yrs required)</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            radar_fig = create_radar_chart(results)
            if radar_fig:
                st.plotly_chart(radar_fig, use_container_width=True)
        
        with chart_col2:
            skill_fig = create_skill_comparison_chart(results['resume_skills'], results['jd_skills'])
            if skill_fig:
                st.plotly_chart(skill_fig, use_container_width=True)
        
        # Skills analysis
        st.subheader("🔧 Skills Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**✅ Skills You Have**")
            if results['matched_skills']:
                for skill in results['matched_skills']:
                    st.markdown(f'<span class="skill-tag">{skill}</span>', unsafe_allow_html=True)
            else:
                st.info("No matching skills found")
        
        with col2:
            st.write("**🔍 Skills Needed**")
            if results['skill_gaps']:
                for skill in results['skill_gaps']:
                    st.markdown(f'<span class="gap-tag">{skill}</span>', unsafe_allow_html=True)
            else:
                st.success("No skill gaps! You have all the required skills")
        
        # Recommendations
        st.subheader("💡 Recommendations")
        
        for rec in results['recommendations']:
            st.markdown(f"""
            <div class="recommendation-box">
                <p>{rec}</p>
            </div>
            """, unsafe_allow_html=True)
        
        if not results['recommendations']:
            st.success("Great job! Your resume is well-matched to this position.")
        
        # Detailed skills breakdown
        with st.expander("View detailed skills breakdown"):
            jd_skills_df = pd.DataFrame({
                'Skill': results['jd_skills'],
                'Status': ['Matched' if skill in results['resume_skills'] else 'Missing' for skill in results['jd_skills']]
            })
            
            st.dataframe(jd_skills_df, use_container_width=True)
    
    elif st.session_state.evaluation_results and st.session_state.evaluation_results.get('error'):
        st.error(f"Error during evaluation: {st.session_state.evaluation_results['error']}")

if __name__ == "__main__":
    main()