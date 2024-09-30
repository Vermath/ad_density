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

def evaluate_ad_density_and_count(image_path):
    # Read and encode the image in base64
    with open(image_path, "rb") as image_file:
        img_data = image_file.read()
        img_b64 = base64.b64encode(img_data).decode('utf-8')
    img_type = mimetypes.guess_type(image_path)[0] or 'image/png'

    # Define the prompt
    prompt = (
        "Evaluate the ad density of the website based on the screenshot. "
        "Return:\n"
        "Ad Density Score: low, medium, or high\n"
        "Explanation: Provide a full explanation of why you assigned that score.\n"
        "Ad Count: The total number of ads present in the screenshot.\n"
        "Count Explanation: Provide a full explanation of how you determined the number of ads.\n"
        "Please format your response exactly as above."
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

    # Extract the response text
    response_text = response.choices[0].message.content.strip()
    return response_text

def extract_score(score_text):
    possible_scores = ["low", "medium", "high"]
    score_text_lower = score_text.lower()
    for score in possible_scores:
        if score in score_text_lower:
            return f"${score}$"
    return None

def parse_response(response_text):
    # Initialize variables
    ad_density_score = None
    ad_density_explanation = None
    ad_count = None
    ad_count_explanation = None

    # Use regex to extract each section
    score_match = re.search(r'Ad Density Score:\s*(.*?)\n', response_text, re.DOTALL)
    explanation_match = re.search(r'Explanation:\s*(.*?)\nAd Count:', response_text, re.DOTALL)
    count_match = re.search(r'Ad Count:\s*(\d+)', response_text)
    count_explanation_match = re.search(r'Count Explanation:\s*(.*)', response_text, re.DOTALL)

    if score_match:
        ad_density_score = score_match.group(1).strip()
    if explanation_match:
        ad_density_explanation = explanation_match.group(1).strip()
    if count_match:
        ad_count = int(count_match.group(1))
    if count_explanation_match:
        ad_count_explanation = count_explanation_match.group(1).strip()

    return ad_density_score, ad_density_explanation, ad_count, ad_count_explanation

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

    # Instructions
    st.markdown("""
    ## Instructions

    - **File Naming:** Ensure that the screenshots in both zip files are named identically. For example, if you have a screenshot named `example_site.png` in the 'before' zip, there should be a screenshot with the same name in the 'after' zip.
    - **Zip Files:** Only upload zip files named appropriately for 'before' and 'after' evaluations.
    - **Upload Files:**
        - Use the file uploaders below to upload your 'before' and 'after' zip files.
    - **Submit:**
        - Click the **Submit** button when you're ready to start the evaluation.
        - Alternatively, check the **Submit as soon as the upload is finished** checkbox if you want the evaluation to start automatically after both files are uploaded.
    - **Disclaimer:**
        - Please note that AI models can sometimes be incorrect. Always use caution and apply common sense when interpreting the results.
    """)

    # File uploaders for the 'before' and 'after' zip files
    before_zip = st.file_uploader("Upload 'Before' Zip File", type='zip')
    after_zip = st.file_uploader("Upload 'After' Zip File", type='zip')

    # Auto-submit checkbox
    auto_submit = st.checkbox("Submit as soon as the upload is finished")

    # Submit button
    submit_button = st.button("Submit")

    # Condition to start processing
    if before_zip and after_zip and (auto_submit or submit_button):
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

            progress_bar = st.progress(0)
            total_sites = len(all_sites)
            current_site = 0

            for site in all_sites:
                before_score = None
                after_score = None
                before_explanation = None
                after_explanation = None
                before_count = None
                after_count = None
                before_count_explanation = None
                after_count_explanation = None

                if site in before_sites:
                    st.write(f"Evaluating 'before' screenshot for {site}...")
                    response_text = evaluate_ad_density_and_count(before_sites[site])
                    # Parse the response
                    ad_density_score, ad_density_explanation, ad_count, ad_count_explanation = parse_response(response_text)
                    before_score = extract_score(ad_density_score)
                    before_explanation = ad_density_explanation
                    before_count = ad_count
                    before_count_explanation = ad_count_explanation

                if site in after_sites:
                    st.write(f"Evaluating 'after' screenshot for {site}...")
                    response_text = evaluate_ad_density_and_count(after_sites[site])
                    # Parse the response
                    ad_density_score, ad_density_explanation, ad_count, ad_count_explanation = parse_response(response_text)
                    after_score = extract_score(ad_density_score)
                    after_explanation = ad_density_explanation
                    after_count = ad_count
                    after_count_explanation = ad_count_explanation

                results.append({
                    'Site': site,
                    'Before Score': before_score,
                    'After Score': after_score,
                    'Before Explanation': before_explanation,
                    'After Explanation': after_explanation,
                    'Before Count': before_count,
                    'After Count': after_count,
                    'Before Count Explanation': before_count_explanation,
                    'After Count Explanation': after_count_explanation
                })

                current_site += 1
                progress_bar.progress(current_site / total_sites)

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
