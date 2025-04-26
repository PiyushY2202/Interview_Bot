import streamlit as st
import os
import json
from typing import Dict, List, Tuple
from datetime import datetime
from langchain_community.llms import Ollama  # For local LLMs like Mistral/LLaMA2
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
from fpdf import FPDF

# Set up the page
st.set_page_config(page_title="AI Interview Coach", page_icon="🤖", layout="wide")

# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'current_question' not in st.session_state:
    st.session_state.current_question = None
if 'interview_type' not in st.session_state:
    st.session_state.interview_type = None
if 'role' not in st.session_state:
    st.session_state.role = None
if 'domain' not in st.session_state:
    st.session_state.domain = None
if 'feedback' not in st.session_state:
    st.session_state.feedback = []
if 'scores' not in st.session_state:
    st.session_state.scores = []
if 'question_count' not in st.session_state:
    st.session_state.question_count = 0
if 'llm_choice' not in st.session_state:
    st.session_state.llm_choice = "Mistral"
if 'interview_history' not in st.session_state:
    st.session_state.interview_history = []

# Initialize LLM
def get_llm():
    if st.session_state.llm_choice == "Mistral":
        return Ollama(model="mistral")
    elif st.session_state.llm_choice == "LLaMA2":
        return Ollama(model="llama2")
    else:
        raise ValueError("Unsupported LLM choice")

# Load configuration
with open('config.json') as f:
    config = json.load(f)

# Add more comprehensive question types
QUESTION_TYPES = {
    "Technical": {
        "Software Engineer": ["Algorithms", "System Design", "Debugging", "Code Review"],
        "Data Scientist": ["ML Concepts", "Data Processing", "Statistics", "Case Studies"],
        "Product Manager": ["Product Sense", "Metrics", "Technical Understanding", "Execution"]
    },
    "Behavioral": {
        "All": ["Teamwork", "Leadership", "Conflict Resolution", "Failure Analysis"]
    }
}

def generate_question(role: str, domain: str, interview_type: str, question_type: str = None) -> str:
    """Generate an interview question with more context"""
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_template(
        """You are an expert interviewer for {role} roles with {domain} specialization.
        Generate a {interview_type} interview question about {question_type} that would properly assess candidates.
        The question should be challenging but fair for mid-level professionals.
        
        Format: <question>\n<context> (brief explanation of what makes a good answer)
        """
    )
    
    if not question_type:
        question_type = "general topics" if interview_type == "Behavioral" else f"{domain} specialization"
    
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "role": role,
        "domain": domain,
        "interview_type": interview_type,
        "question_type": question_type
    })

def evaluate_response(question: str, response: str, interview_type: str) -> Tuple[str, int]:
    """More sophisticated evaluation with detailed rubrics"""
    llm = get_llm()
    
    rubric = {
        "Technical": {
            "criteria": ["Accuracy", "Depth", "Problem-solving", "Communication"],
            "weights": [0.4, 0.3, 0.2, 0.1]
        },
        "Behavioral": {
            "criteria": ["STAR Format", "Relevance", "Impact", "Self-awareness"],
            "weights": [0.3, 0.2, 0.3, 0.2]
        }
    }[interview_type]
    
    prompt = ChatPromptTemplate.from_template(
        """Evaluate this interview response based on the following rubric:
        Criteria: {criteria}
        Weights: {weights}
        
        Question: {question}
        Response: {response}
        
        Provide:
        1. Detailed feedback highlighting strengths and specific improvement suggestions
        2. Scores for each criterion (1-5 scale)
        3. Overall weighted score (1-10 scale)
        
        Format:
        Feedback: <feedback>
        Scores: <criterion>:<score>,...
        Overall: <overall>/10
        """
    )
    
    evaluation = (prompt | llm | StrOutputParser()).invoke({
        "question": question,
        "response": response,
        "criteria": ", ".join(rubric["criteria"]),
        "weights": ", ".join(map(str, rubric["weights"]))
    })
    
    # Parse the evaluation
    feedback = evaluation.split("Feedback:")[1].split("Scores:")[0].strip()
    scores = dict(item.split(":") for item in evaluation.split("Scores:")[1].split("Overall:")[0].strip().split(","))
    overall = float(evaluation.split("Overall:")[1].strip().split("/")[0])
    
    return feedback, scores, overall

def generate_pdf_report():
    """Generate a PDF report of the interview session"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.cell(200, 10, txt="Interview Performance Report", ln=1, align="C")
    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d')}", ln=1)
    pdf.cell(200, 10, txt=f"Role: {st.session_state.role}", ln=1)
    pdf.cell(200, 10, txt=f"Domain: {st.session_state.domain}", ln=1)
    pdf.cell(200, 10, txt=f"Interview Type: {st.session_state.interview_type}", ln=1)
    
    pdf.ln(10)
    pdf.cell(200, 10, txt="Session Summary", ln=1)
    
    if st.session_state.question_count > 0:
        avg_score = sum(st.session_state.scores) / len(st.session_state.scores)
        pdf.cell(200, 10, txt=f"Average Score: {avg_score:.1f}/10", ln=1)
        
        pdf.ln(5)
        pdf.cell(200, 10, txt="Question-by-Question Feedback:", ln=1)
        
        for i, (q, a, f, s) in enumerate(st.session_state.conversation, 1):
            pdf.multi_cell(0, 10, txt=f"Q{i}: {q}\nYour Answer: {a}\nFeedback: {f}\nScore: {s}/10\n")
            pdf.ln(2)
    
    # Save to file
    filename = f"interview_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    pdf.output(filename)
    return filename

# Main app interface
st.title("🤖 AI Interview Coach")
st.markdown("Practice technical and behavioral interviews with AI feedback")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # LLM Selection
    st.session_state.llm_choice = st.radio(
        "Choose AI Model:",
        ["Mistral", "LLaMA2"],
        index=0,
        help="Mistral is better for technical interviews, LLaMA2 for behavioral"
    )
    
    # Only allow configuration if no interview in progress
    if st.session_state.question_count == 0:
        st.session_state.role = st.selectbox(
            "Target Role:",
            config["roles"],
            index=0
        )
        
        st.session_state.domain = st.selectbox(
            "Specialization:",
            config["domains"],
            index=0
        )
        
        st.session_state.interview_type = st.radio(
            "Interview Type:",
            ["Technical", "Behavioral"],
            index=0
        )
        
        # Question type selection
        if st.session_state.interview_type == "Technical":
            q_type = st.selectbox(
                "Question Focus:",
                QUESTION_TYPES["Technical"][st.session_state.role]
            )
        else:
            q_type = st.selectbox(
                "Behavioral Area:",
                QUESTION_TYPES["Behavioral"]["All"]
            )
        
        if st.button("Start Interview"):
            st.session_state.current_question = generate_question(
                st.session_state.role,
                st.session_state.domain,
                st.session_state.interview_type,
                q_type
            )
            st.rerun()
    else:
        st.warning("Interview in progress. Finish or reset to change settings.")
        if st.button("Reset Interview"):
            st.session_state.interview_history.append({
                "date": datetime.now(),
                "role": st.session_state.role,
                "type": st.session_state.interview_type,
                "score": sum(st.session_state.scores)/len(st.session_state.scores) if st.session_state.scores else 0,
                "questions": st.session_state.question_count
            })
            st.session_state.conversation = []
            st.session_state.current_question = None
            st.session_state.feedback = []
            st.session_state.scores = []
            st.session_state.question_count = 0
            st.rerun()

# Main interview area
col1, col2 = st.columns([2, 1])

with col1:
    if st.session_state.current_question:
        st.subheader(f"{st.session_state.interview_type} Question")
        
        # Display question with context if available
        if "\n" in st.session_state.current_question:
            question, context = st.session_state.current_question.split("\n", 1)
            st.markdown(f"**{question}**")
            with st.expander("What makes a good answer?"):
                st.info(context)
        else:
            st.markdown(f"**{st.session_state.current_question}**")
        
        response = st.text_area(
            "Your response:", 
            key="response", 
            height=200,
            placeholder="Type your answer here...",
            help="For technical questions, you can include code snippets"
        )
        
        # Response actions
        col1a, col1b, col1c = st.columns(3)
        with col1a:
            if st.button("Submit", type="primary"):
                if response.strip():
                    feedback, scores, overall = evaluate_response(
                        st.session_state.current_question,
                        response,
                        st.session_state.interview_type
                    )
                    
                    st.session_state.conversation.append(
                        (st.session_state.current_question, response, feedback, overall)
                    )
                    st.session_state.feedback.append(feedback)
                    st.session_state.scores.append(overall)
                    st.session_state.question_count += 1
                    
                    # Generate next question
                    st.session_state.current_question = generate_question(
                        st.session_state.role,
                        st.session_state.domain,
                        st.session_state.interview_type
                    )
                    st.rerun()
                else:
                    st.warning("Please enter a response before submitting.")
        
        with col1b:
            if st.button("Skip Question"):
                st.session_state.current_question = generate_question(
                    st.session_state.role,
                    st.session_state.domain,
                    st.session_state.interview_type
                )
                st.rerun()
        
        with col1c:
            if st.button("End Interview"):
                st.session_state.current_question = None
                st.rerun()
        
        # Display previous Q&A if any
        if st.session_state.conversation:
            st.subheader("Your Responses")
            for i, (q, a, f, s) in enumerate(st.session_state.conversation, 1):
                with st.expander(f"Question {i} | Score: {s:.1f}/10", expanded=False):
                    st.markdown(f"**Question:** {q.split('\n')[0]}")
                    st.markdown(f"**Your Answer:** {a}")
                    st.markdown(f"**Feedback:** {f}")

    elif st.session_state.question_count > 0:
        st.subheader("Interview Summary")
        
        avg_score = sum(st.session_state.scores)/len(st.session_state.scores)
        st.metric("Average Score", f"{avg_score:.1f}/10")
        
        st.subheader("Strengths")
        strengths = [f for f in st.session_state.feedback if any(word in f.lower() for word in ["good", "strong", "excellent", "well"])]
        for strength in strengths[:3]:  # Show top 3 strengths
            st.success(strength)
        
        st.subheader("Areas for Improvement")
        improvements = [f for f in st.session_state.feedback if any(word in f.lower() for word in ["improve", "better", "weak", "could"])]
        for improvement in improvements[:3]:  # Show top 3 improvements
            st.warning(improvement)
        
        # Generate recommendations
        llm = get_llm()
        prompt = ChatPromptTemplate.from_template(
            """Based on these interview feedback points:
            {feedback}
            
            Provide 3-5 specific recommendations for improvement for a {role} 
            with {domain} specialization focusing on {interview_type} interviews.
            Format as bullet points.
            """
        )
        
        recommendations = (prompt | llm | StrOutputParser()).invoke({
            "feedback": "\n".join(st.session_state.feedback),
            "role": st.session_state.role,
            "domain": st.session_state.domain,
            "interview_type": st.session_state.interview_type
        })
        
        st.subheader("Recommended Next Steps")
        st.markdown(recommendations)
        
        # Export options
        st.download_button(
            label="Download PDF Report",
            data=generate_pdf_report(),
            file_name=f"interview_report_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf"
        )
        
        if st.button("Start New Interview"):
            st.session_state.interview_history.append({
                "date": datetime.now(),
                "role": st.session_state.role,
                "type": st.session_state.interview_type,
                "score": avg_score,
                "questions": st.session_state.question_count
            })
            st.session_state.conversation = []
            st.session_state.feedback = []
            st.session_state.scores = []
            st.session_state.question_count = 0
            st.rerun()
    
    else:
        st.markdown("""
        ## Welcome to AI Interview Coach!
        
        **Practice makes perfect** - get ready for your next job interview with AI-powered coaching.
        
        ### How it works:
        1. Select your target role and specialization
        2. Choose between technical or behavioral interviews
        3. Answer questions and get immediate feedback
        4. Review your performance and improve
        
        ### Features:
        - Role-specific questions
        - Detailed scoring rubrics
        - Personalized feedback
        - Exportable reports
        - Multiple AI model options
        
        Get started by configuring your interview in the sidebar.
        """)

with col2:
    if st.session_state.question_count > 0:
        st.subheader("Performance Metrics")
        
        # Score progression chart
        if len(st.session_state.scores) > 1:
            st.line_chart(
                pd.DataFrame({
                    "Question": range(1, len(st.session_state.scores)+1),
                    "Score": st.session_state.scores
                }).set_index("Question"),
                height=200
            )
        
        # Current score
        if st.session_state.scores:
            latest_score = st.session_state.scores[-1]
            st.metric("Latest Score", f"{latest_score:.1f}/10")
        
        # Feedback highlights
        if st.session_state.feedback:
            with st.expander("Latest Feedback"):
                st.write(st.session_state.feedback[-1])
    
    # Interview history
    if st.session_state.interview_history:
        st.subheader("Your History")
        history_df = pd.DataFrame(st.session_state.interview_history)
        st.dataframe(
            history_df.sort_values("date", ascending=False).head(5),
            hide_index=True,
            column_config={
                "date": "Date",
                "role": "Role",
                "type": "Type",
                "score": st.column_config.NumberColumn("Score", format="%.1f/10"),
                "questions": "Questions"
            }
        )