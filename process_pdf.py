# process_pdf.py

import os
from dotenv import load_dotenv
import logging
import openai
import PyPDF2
import re
import numpy as np
import faiss
import streamlit as st
import pickle

# Import necessary classes from langchain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load environment variables
load_dotenv()
openai_api_key = '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'

if not openai_api_key:
    raise ValueError("OpenAI API key not found. Please set it in the .env file.")

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize OpenAI API
openai.api_key = openai_api_key

# Initialize the chat model
llm = ChatOpenAI(temperature=0.7)

# Define the prompt template for scenario generation
scenario_template = PromptTemplate(
    input_variables=['branch', 'role', 'situation', 'difficulty'],
    template=(
        "Kirjoita skenaario toisessa persoonassa siten, että voin vastata ja kertoa, kuinka toimisin tilanteessa.\n"
        "Skenaarion tulee testata minun johtamistaitojani ja tarjota mahdollisuus soveltaa Johtajan Käsikirja 2022:n 'syväjohtamisen' oppeja käytännössä. \n"
        "Kirjoita skenaario vain suomeksi. Roolina tulee olla '{role}' aselajina '{branch}'. Kirjoita tilanne toisessa persoonassa ja varmista, että se sopii rooliin ja aselajin toimintaan. \n"
        "Vaikeustaso vaikuttaa tilanteen seuraaviin osa-alueisiin:\n"
        "- **Arkinen**: Tilanne sisältää yhden selkeän ongelman, joukkueen moraali ja resurssit ovat hyvässä kunnossa, ja tehtävän tavoitteet ovat suoraviivaisia. Roolin vastuut ja ympäristön vaatimukset ovat kohtuullisia.\n"
        "- **Keskivaikea**: Tilanne sisältää useita samanaikaisia haasteita, kuten resurssipulaa, ristiriitaisia tavoitteita tai epävarmuutta tiedoissa. Moraalin ylläpito ja päätöksenteko ovat keskiössä.\n"
        "- **Haastava**: Tilanne on monimutkainen ja nopeatempoinen, sisältää korkean riskin päätöksiä ja merkittäviä seurauksia epäonnistumiselle. Resurssit ovat kriittisen rajalliset, ja tehtävän onnistuminen vaatii luovaa ongelmanratkaisua ja erinomaisia johtamistaitoja.\n"
        "Tässä tapauksessa valittu vaikeustaso on '{difficulty}'.\n"
        "Skenaarion on kuvattava selkeästi seuraavat osiot suomeksi:\n"
        "- **Tiedot ja tilannekuva**: Tilanne on {situation}. Missä tilanne tapahtuu? Ketkä ovat mukana? Mitä tilanteessa tapahtuu? Luo vähintään kolmen kappaleen pituinen tarina.\n"
        "Mukauta tämä aselajin erityispiirteisiin ja roolin vastuisiin.\n"
        "- **Tavoite:**: Tiivistä yhteen lauseeseen, mitä mikä on lopputulos joka minun tulisi saavuttaa? Esim. 'Selvitä ryhmän jäsenten välinen konflikti.'\n"
        "\n"
        "Muista varmistaa, että:\n"
        "- Skenaario sopii annetulle roolille ja aselajille.\n"
        "- Tehtävän haasteet eivät ole yleisluontoisia, vaan aselajin ja roolin kannalta erityisiä.\n"
        "- Taktiselle osaamiselle ei anneta liikaa painoarvoa; pääpaino on johtajuuden ja päätöksenteon arvioinnissa.\n"
        "Kirjoita 'Tiedot ja tilannekuva' noin kolmen kappaleen mittaisena ja sisällytä riittävästi yksityiskohtia, jotta käyttäjä voi tehdä päätöksiä skenaarion pohjalta. Älä kirjoita toimintaohjeita, vaan anna minun itse keksiä kuinka toimin."
    )
)




# Define the prompt template for feedback generation
feedback_template = PromptTemplate(
    input_variables=['scenario', 'user_response', 'relevant_texts'],
    template=(
        "Give the user feedback in Finnish."
        "Focus on analyzing my actions and that the feedback focuses on my actions and decisions. "
        "The feedback should focus on my skills 'johtaminen' (management -> resources, tasks, risks, time) and 'johtajuus' (leadership -> managing people, their morale and depending on the situation, developing subordinates.)."
        "FEEDBACK: Think of an answer to these questions: What did I do well? What could I improve on? How could I improve on these areas?\n\n"
        "**Scenario**:\n{scenario}\n\n"
        "**What I would have done**:\n{user_response}\n\n"
        "**Johtajan käsikirjan periaatteet: {relevant_texts}**\n\n\n"
        "Anna palautetta seuraavassa muodossa:\n"
        "- **Näin tapahtui**: Write a part two for the scenario, based on how things would've likely turned out after the user's actions. This must align with your feedback.\n"
        "- **Vahvuudet**: Tell me what I performed well at, if I did perform well at anything. If I performed badly or did not respond to the situation, you can leave this field blank\n"
        "- **Kehityskohteet**: What did I do wrong, forget to do or do poorly? Make sure that this does not conflict with what you wrote in 'Vahvuudet'. \n"
        "- **Suositukset**: Give clear and actionable recommendations, how could I achieve the improvements you listed in 'kehityskohteet'\n\n"
        "**''Example quote'' - Johtajan Käsikirja 2022** Below 'suositukset', add a word to word quote from Johtajan Käsikirja 2022, that both supports your feedback and fits the scenario.\n"
        "If you think my response is a joke, dangerously bad, or completely irrelevant to the situation, point this out in the feedback."
    )
)



# Create the LLMChain for scenario generation
scenario_chain = LLMChain(llm=llm, prompt=scenario_template)

# Create the LLMChain for feedback generation
feedback_chain = LLMChain(llm=llm, prompt=feedback_template)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        logging.error(f"Error reading PDF file: {e}")
        return ""
    
# Function to parse the PDF into hierarchical structure
def parse_pdf_to_structure(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text()
            
        # Use regular expressions to identify chapters, sections, and subsections
        chapters = re.split(r'(?<=\n)(OSIO \d+ .*?)(?=\n)', text)
        structured_data = []
        
        for chapter_idx in range(1, len(chapters), 2):  # Skip odd indices for content
            chapter_title = chapters[chapter_idx].strip()
            chapter_content = chapters[chapter_idx + 1]
            sections = re.split(r'(?<=\n)(\d+ .*?)(?=\n)', chapter_content)

            for section_idx in range(1, len(sections), 2):
                section_title = sections[section_idx].strip()
                section_content = sections[section_idx + 1]
                subsections = re.split(r'(?<=\n)(\d+\.\d+ .*?)(?=\n)', section_content)
                
                for subsection_idx in range(1, len(subsections), 2):
                    subsection_title = subsections[subsection_idx].strip()
                    subsection_content = subsections[subsection_idx + 1]
                    
                    # Store structured data
                    structured_data.append({
                        'chapter': chapter_title,
                        'section': section_title,
                        'subsection': subsection_title,
                        'content': subsection_content.strip()
                    })
        return structured_data
    except Exception as e:
        logging.error(f"Error parsing PDF: {e}")
        return []

# Function to clean and split text into chunks
def clean_and_split_text(text, max_chunk_size=500):
    # Remove unnecessary whitespace
    text = re.sub('\s+', ' ', text)
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ''
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += ' ' + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def truncate_text(text, max_length=8191):
    if len(text) > max_length:
        logging.warning(f"Truncating text to {max_length} characters.")
        return text[:max_length]
    return text

def generate_embedding(text):
    try:
        if not text or not text.strip():
            logging.error("Invalid input: Text is empty or whitespace only.")
            return None

        # Log input for debugging
        logging.info(f"Generating embedding for text: {text[:100]}... [truncated]")

        # Call OpenAI API
        response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=text.strip()
        )

        # Safely access embedding
        embedding = response.data[0].embedding

        if embedding is None:
            raise ValueError("Embedding not found in response.")
        
        return np.array(embedding, dtype='float32')

    except ValueError as e:
        logging.error(f"Value error: {e}")
        return None
    except Exception as e:
        logging.error(f"Error generating embedding: {e}")
        return None





# Function to create embeddings store
def create_embeddings_store(chunks):
    embeddings = []
    for idx, chunk in enumerate(chunks):
        embedding = generate_embedding(chunk)
        if embedding is not None:
            embeddings.append({
                'id': idx,
                'text': chunk,
                'embedding': embedding
            })
    return embeddings

# Function to build FAISS index
def build_faiss_index(embeddings_store):
    if not embeddings_store:
        raise ValueError("Embeddings store is empty. Cannot build FAISS index.")
    dimension = len(embeddings_store[0]['embedding'])
    index = faiss.IndexFlatL2(dimension)
    # Prepare the embeddings matrix
    embeddings_matrix = np.array([item['embedding'] for item in embeddings_store])
    index.add(embeddings_matrix)
    return index, embeddings_matrix

# Function to search for similar texts
def search_similar_texts(user_embedding, embeddings_store, index, top_k=5):
    user_embedding = user_embedding.reshape(1, -1)
    distances, indices = index.search(user_embedding, top_k)
    similar_texts = []
    for idx in indices[0]:
        similar_texts.append(embeddings_store[idx]['text'])
    return similar_texts

# Function to generate scenario
def generate_scenario(branch, role, situation, difficulty):
    try:
        scenario = scenario_chain.run(branch=branch, role=role, situation=situation, difficulty=difficulty)
        return scenario.strip()
    except Exception as e:
        logging.error(f"Error generating scenario: {e}")
        return "An error occurred while generating the scenario."

# Function to generate feedback with verified quotes
def generate_feedback(scenario, user_response, relevant_texts):
    knowledge_base = "\n".join([f"{text['content']} (Osio: {text['chapter']}, Kappale: {text['section']})"
                                 for text in relevant_texts])

    try:
        feedback = feedback_chain.run(
            scenario=scenario,
            user_response=user_response,
            relevant_texts=knowledge_base
        )
        return feedback.strip()
    except Exception as e:
        logging.error(f"Error generating feedback: {e}")
        return "An error occurred while generating feedback."
    
# Function to search for similar texts
def search_similar_texts(user_embedding, embeddings_store, index, top_k=5):
    user_embedding = user_embedding.reshape(1, -1)
    distances, indices = index.search(user_embedding, top_k)
    similar_texts = []

    for idx in indices[0]:
        if idx >= len(embeddings_store) or idx < 0:  # Ensure index is valid
            logging.error(f"Invalid index {idx} returned from FAISS search.")
            continue

        if 'text' not in embeddings_store[idx]:  # Ensure 'text' exists
            logging.error(f"Entry at index {idx} is missing 'text'.")
            continue

        similar_texts.append(embeddings_store[idx]['text'])

    return similar_texts


# Function to retrieve relevant sections
def retrieve_relevant_sections(user_response, embeddings_store, index, top_k=5):
    user_embedding = generate_embedding(user_response)
    if user_embedding is None:
        logging.error("Failed to generate embedding for user response.")
        return []

    distances, indices = index.search(user_embedding.reshape(1, -1), top_k)
    relevant_sections = []

    for idx in indices[0]:
        if idx >= len(embeddings_store) or idx < 0:
            logging.error(f"Invalid index {idx} returned from FAISS search.")
            continue

        entry = embeddings_store[idx]
        relevant_sections.append({
            'chapter': entry.get('chapter', ''),
            'section': entry.get('section', ''),
            'subsection': entry.get('subsection', ''),
            'content': entry.get('content', ''),
        })

    return relevant_sections

# Function to load embeddings and index from disk
@st.cache_resource
def load_embeddings_and_index(embeddings_path, index_path):
    # Load embeddings_store
    with open(embeddings_path, 'rb') as f:
        embeddings_store = pickle.load(f)
    
    # Load FAISS index
    index = faiss.read_index(index_path)
    
    return embeddings_store, index

# Main execution
def main():
    # Path to your PDF file
    pdf_path = 'JOHT_KK_22.pdf'  # Replace with your actual PDF file path

    # Extract text from PDF
    logging.info("Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)

    if not text:
        logging.error("No text extracted from PDF.")
        return

    # Clean and split text into chunks
    logging.info("Processing text...")
    chunks = clean_and_split_text(text)

    # Create embeddings store
    logging.info("Generating embeddings...")
    embeddings_store = create_embeddings_store(chunks)

    if not embeddings_store:
        logging.error("No embeddings generated.")
        return

    # Build FAISS index
    logging.info("Building FAISS index...")
    index, embeddings_matrix = build_faiss_index(embeddings_store)


    # Example usage:
    # Generate a scenario
    branch = 'Logistics'
    role = 'Team Leader'
    situation = 'Accident'
    difficulty = 'Medium'

    logging.info("Generating scenario...")
    scenario = generate_scenario(branch, role, situation, difficulty)
    print("Scenario:")
    print(scenario)
    print()

    # Get user response (for testing purposes, we can input manually)
    user_response = input("Enter your response to the scenario:\n")

    # Generate embedding for user response
    logging.info("Generating embedding for user response...")
    user_embedding = generate_embedding(user_response)

    if user_embedding is None:
        logging.error("Failed to generate embedding for user response.")
        return

    # Retrieve relevant handbook sections
    logging.info("Retrieving relevant handbook sections...")
    relevant_texts = search_similar_texts(user_embedding, embeddings_store, index)

    # Generate feedback
    logging.info("Generating feedback...")
    feedback = generate_feedback(scenario, user_response, relevant_texts)
    print("\nFeedback:")
    print(feedback)

if __name__ == '__main__':
    main()
