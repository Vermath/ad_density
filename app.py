import streamlit as st
import zipfile
import tempfile
import os
import base64
import mimetypes
import pandas as pd
from openai import OpenAI
import re

# Initialize OpenAI client using Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def evaluate_ad_density(image_path):
    # Read and encode the image in base64
    with open(image_path, "rb") as image_file:
        img_data = image_file.read()
        img_b64 = base64.b64encode(img_data).decode('utf-8')
    img_type = mimetypes.guess_type(image_path)[0] or 'image/png'

    # Define the prompt
    prompt = (
        "Evaluate the ad density of the website based on the screenshot. "
        "Return a score of low, medium, or high, and provide a full explanation "
        "of why you assigned that score."
    )

    # Create the completion
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{img_type};base64,{img_b64}"},
                    },
                ],
            }
        ],
    )

    # Extract the response text (full explanation)
    response_text = response.choices[0].message.content.strip()
    return response_text

def extract_score(explanation):
    possible_scores = ["low", "medium", "high"]
    explanation_lower = explanation.lower()
    score_found = None
    for score in possible_scores:
        # Use word boundaries to avoid partial matches
        if re.search(r'\b' + re.escape(score) + r'\b', explanation_lower):
            score_found = score
            break
    if score_found:
        return f"${score_found}$"
    else:
        return None

def get_image_files(directory):
    image_files = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the file is an image
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                site_name = os.path.splitext(file)[0]
                image_files[site_name] = os.path.join(root, file)
    return image_files

def main():
    st.title("Ad Density Evaluation Tool")

    # File uploaders for the 'before' and 'after' zip files
    before_zip = st.file_uploader("Upload 'Before' Zip File", type='zip')
    after_zip = st.file_uploader("Upload 'After' Zip File", type='zip')

    if before_zip and after_zip:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract 'before' zip file
            before_dir = os.path.join(tmpdir, 'before')
            os.makedirs(before_dir, exist_ok=True)
            with zipfile.ZipFile(before_zip, 'r') as zip_ref:
                zip_ref.extractall(before_dir)

            # Extract 'after' zip file
            after_dir = os.path.join(tmpdir, 'after')
            os.makedirs(after_dir, exist_ok=True)
            with zipfile.ZipFile(after_zip, 'r') as zip_ref:
                zip_ref.extractall(after_dir)

            # Get image files from the directories
            before_sites = get_image_files(before_dir)
            after_sites = get_image_files(after_dir)
            all_sites = set(before_sites.keys()).union(after_sites.keys())

            results = []

            for site in all_sites:
                before_score = None
                after_score = None
                before_explanation = None
                after_explanation = None

                if site in before_sites:
                    st.write(f"Evaluating 'before' screenshot for {site}...")
                    before_explanation = evaluate_ad_density(before_sites[site])
                    before_score = extract_score(before_explanation)

                if site in after_sites:
                    st.write(f"Evaluating 'after' screenshot for {site}...")
                    after_explanation = evaluate_ad_density(after_sites[site])
                    after_score = extract_score(after_explanation)

                results.append({
                    'Site': site,
                    'Before Score': before_score,
                    'After Score': after_score,
                    'Before Explanation': before_explanation,
                    'After Explanation': after_explanation
                })

            # Create a DataFrame and display it
            df = pd.DataFrame(results)
            st.dataframe(df)

            # Provide a CSV download
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name='ad_density_results.csv',
                mime='text/csv'
            )

if __name__ == '__main__':
    main()
