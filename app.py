# app.py

import os
from dotenv import load_dotenv
import logging
import streamlit as st
import pickle
import faiss
import openai

# Import necessary classes from langchain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Import necessary functions from process_pdf.py
from process_pdf import (
    generate_scenario,
    generate_feedback,
    generate_embedding,
    search_similar_texts,
    retrieve_relevant_sections,
    load_embeddings_and_index  # Assuming you have this function
)

# Load environment variablesQ
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

if not openai_api_key:
    raise ValueError("OpenAI API key not found. Please set it in the .env file.")

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize OpenAI API
openai.api_key = openai_api_key

# Initialize the chat model
llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.7)

# Define prompt templates and chains if not already defined in process_pdf.py

# Scenario generation prompt template
scenario_template = PromptTemplate(
    input_variables=['branch', 'role', 'situation', 'difficulty'],
    template=(
        "You are a military training simulator. Generate a {difficulty} difficulty leadership scenario "
        "for a {role} in the {branch} branch during a {situation} situation. "
        "The scenario should challenge the user's leadership skills and be less than 150 words."
    )
)

scenario_chain = LLMChain(llm=llm, prompt=scenario_template)

# Feedback generation prompt template
feedback_template = PromptTemplate(
    input_variables=['scenario', 'user_response', 'relevant_texts'],
    template=(
        "As an expert military leadership instructor, provide detailed feedback on the user's response.\n\n"
        "**Scenario:**\n{scenario}\n\n"
        "**User's Response:**\n{user_response}\n\n"
        "**Relevant Principles from the Handbook:**\n{relevant_texts}\n\n"
        "Provide feedback structured as follows:\n"
        "- **Strengths**: List what the user did well.\n"
        "- **Areas for Improvement**: List specific areas where the user can improve. Make sure to not contradict what you wrote in strengths\n"
        "- **Recommendations**: Provide actionable advice based on the handbook principles."
    )
)

feedback_chain = LLMChain(llm=llm, prompt=feedback_template)

def main():
    st.logo("practilearn_logo.png", size="large")
    st.image("practilearn_logo.png", width=200)
    st.title("PraxiLearn.AI (V0.1.0)")

    # Step 1: User selects options
    st.header("Asetukset")
    branch = st.selectbox('Aselaji', ['Jääkäri', 'Sotilaspoliisi', 'Viestintä', 'Huolto', 'Kuljetus', 'Lääkintä', 'Ilmatorjunta', 'Pioneeri', 'Tykistö', 'Esikunta', 'Tiedustelu', 'Sotilaskeittäjä', 'Satunnainen'])
    role = st.selectbox('Rooli', ['Ryhmän jäsen', 'Partionjohtaja', 'Ryhmänjohtaja', 'Joukkueenjohtaja', 'Komppanianvarapäällikkö', 'Komppanianpäällikkö'])
    situation = st.selectbox('Tilanteen tarkenne', ['Taistelu', 'Kriisinhallinta', 'Ryhmän sisäinen konflikti', 'Tapaturma/onnettomuus', 'Ongelmatilanne kaluston kanssa', 'Ongelmatilanne siviilien kanssa'])
    difficulty = st.selectbox('Tilanteen haastavuus', ['Arkinen', 'Keskivaikea', 'Haastava'])

    # Load embeddings and index once
    embeddings_store, index = load_embeddings_and_index('embeddings_store_johtkk22.pkl', 'faiss_index.index')

    # Initialize session state variables
    if 'scenario' not in st.session_state:
        st.session_state['scenario'] = ''
    if 'user_response' not in st.session_state:
        st.session_state['user_response'] = ''
    if 'feedback' not in st.session_state:
        st.session_state['feedback'] = ''

    # Step 2: User clicks "Generate"
    if st.button('Aloita'):
        with st.spinner('Luodaan tilannetta...'):
            st.session_state['scenario'] = generate_scenario(
                branch, role, situation, difficulty
            )
            st.session_state['user_response'] = ''  # Reset user response
            st.session_state['feedback'] = ''  # Reset feedback
        st.success('Tilanne valmis!')

    # Step 3: Display scenario and text box for user's answer
    if st.session_state['scenario']:
        st.subheader("Tilanne")
        st.write(st.session_state['scenario'])

        # Step 4: User enters response
        st.subheader("Kuinka toimit?")
        st.session_state['user_response'] = st.text_area(
            "Kuvaile, kuinka toimisit kuvatussa tilanteessa:",
            st.session_state['user_response'],
            height=200
        )

        # Step 5: User clicks "Send"
        if st.button('Lähetä'):
            if not st.session_state['user_response'].strip():
                st.warning('Please enter a response.')
            else:
                with st.spinner('Generating feedback...'):
                    # Retrieve relevant handbook sections
                    relevant_texts = retrieve_relevant_sections(
                        st.session_state['user_response'],
                        embeddings_store,
                        index
                    )
                    # Generate feedback
                    st.session_state['feedback'] = generate_feedback(
                        st.session_state['scenario'],
                        st.session_state['user_response'],
                        relevant_texts
                    )
                st.success('Feedback generated!')

    # Step 5: Display feedback
    if st.session_state['feedback']:
        st.subheader("Palaute")
        st.write(st.session_state['feedback'])

if __name__ == '__main__':
    main()
