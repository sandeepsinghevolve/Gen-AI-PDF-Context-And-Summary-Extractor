import streamlit as st
import os
from PyPDF2 import PdfReader # For reading the PDF
from google.generativeai import GenerativeModel
import google.generativeai as genai
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()  # take environment variables from .env.

os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Loading the Gemini Pro model
model = genai.GenerativeModel('gemini-pro')

# Retrieves the embedded metadata from the PDF file.PDF files can store metadata information such as title, author, and keywords.
# Function will attempts to extract the document's title from the metadata.
def extract_metadata(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    document_metadata = pdf_reader.metadata 
    
    title = document_metadata.get("/Title") 
    author = document_metadata.get("/Author")
    keywords = document_metadata.get("/Keywords")

    return title, author, keywords


# Function to summarize the text/PDF
# The model's output is treated as a condensed version of the original text, highlighting the main ideas.
def summarize_pdf(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    
    summaries = []
    for chunk in chunks:
        prompt = "Summarize the following text and summarize should be within 150 tokens :\n" + chunk
        response = model.generate_content(prompt)
        chunk_summary = response.text.strip()
        summaries.append(chunk_summary)
        
    combined_summary = " ".join(summaries)
    return combined_summary

# Function to categorize the text/pdf on the basis of summary
# Categorization: The output is parsed as a list of categories, representing the primary topics or themes identified in the text.
# If metadata is provided and the first element of metadata is truthy (non-empty), it assigns the first element of metadata to the variable context.
# Metadata can provide concise and accurate information about the document's topic, potentially improving categorization.

def categorize_pdf(text, use_summary=True, metadata=None):
    if use_summary:
        text = summarize_pdf(text)  # Summarize first if needed

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=100)
        chunks = text_splitter.split_text(text)

        categories = []
        for chunk in chunks:
            prompt = "Categorize the following text and categorization max token should be 10:\n" + chunk
            response = model.generate_content(prompt)
            chunk_categories = response.text.strip().splitlines()
            categories.extend(chunk_categories)

        # Combine categories to extract a single context
        unique_categories = list(set(categories))
        combined_categories = []
        for category in unique_categories:
            count = categories.count(category)
            combined_categories.extend([category] * count)  # Add multiple times based on frequency

        most_frequent_category = max(set(combined_categories), key=combined_categories.count)
        context = most_frequent_category

    return context


# Streamlit app structure
st.set_page_config(page_title="PDF Context and Summary Extractor")
st.header("PDF Context and Summary Extractor")
uploaded_file = st.file_uploader("Upload PDF file", type="pdf")

if uploaded_file:
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    title, author, keywords = extract_metadata(uploaded_file)
    st.write("Title:", title)
    st.write("Author:", author)
    if keywords:
        st.write("Keywords:", keywords)

    context = categorize_pdf(text, use_summary=True, metadata=(title, author, keywords))
    st.write("Context:", context)

    # Option to display summary
    if st.button("Show Summary"):
        summary = summarize_pdf(text)
        st.write("Summary:", summary)
