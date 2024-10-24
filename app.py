import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import gradio as gr


df = pd.read_csv('AnalyticsVidhya.csv')

# Load the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 1: Combine the relevant columns to create a single text representation
df['CombinedText'] = df.apply(
    lambda row: (
        f"Title: {str(row['CourseTitle'])}. "
        f"Level: {str(row['Level']) if pd.notnull(row['Level']) else 'Not specified'}. "
        f"Duration: {str(row['Time(Hours)']) if pd.notnull(row['Time(Hours)']) else 'Unknown'} hours. "
        f"Category: {str(row['Category']) if pd.notnull(row['Category']) else 'Not specified'}. "
        f"Description: {str(row['Description']) if pd.notnull(row['Description']) else 'No description available'}"
    ), axis=1)

# Step 2: Generate embeddings for all courses
embeddings = model.encode(df['CombinedText'].tolist(), show_progress_bar=True)

# Step 3: Create FAISS index
embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)

# Add the embeddings to the index
index.add(np.array(embeddings))

# Step 4: Search function to query courses
def search_courses(query, top_k=5):
    # Generate embedding for the query
    query_embedding = model.encode([query])[0]

    # Perform the search in FAISS
    distances, indices = index.search(np.array([query_embedding]), top_k)

    # Retrieve the corresponding course information
    results = []
    for idx in indices[0]:
        course = {
            'CourseTitle': df.iloc[idx]['CourseTitle'],
            'Description': df.iloc[idx]['Description'],
            'Level': df.iloc[idx]['Level'],
            'Category': df.iloc[idx]['Category'],
            'NumberOfLessons': df.iloc[idx]['NumberOfLessons']
        }
        results.append(course)
    
    return results

# Step 5: Define Gradio interface function
def gradio_search(query, top_k):
    results = search_courses(query, top_k)
    display_results = [
        f"Title: {res['CourseTitle']}\n"
        f"Description: {res['Description']}\n"
        f"Category: {res['Category']}\n"
        f"Level: {res['Level']}"
        for res in results
    ]
    return "\n\n".join(display_results)

# Step 6: Create Gradio interface
interface = gr.Interface(
    fn=gradio_search,
    inputs=[
        gr.Textbox(label="Search in our courses"),
        gr.Slider(minimum=1, maximum=10, step=1, value=5, label="Number of Results")
    ],
    outputs="text",
    title="Smart Course Search",
    description="Analytics Vidhya Free Courses",
    flagging_mode=None  # Disable flagging button
)

# Step 7: Launch the Gradio interface
interface.launch()
