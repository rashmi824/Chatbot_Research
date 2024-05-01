import os
from PyPDF2 import PdfReader

def extract_text_from_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
        return None

def extract_text_from_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading text from {file_path}: {e}")
        return None

def merge_text_files(folder_path, output_file_path):
    success_count = 0
    failure_count = 0

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for root, _, files in os.walk(folder_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                if file_path.lower().endswith('.pdf'):
                    print(f"Extracting text from {file_path}...")
                    text = extract_text_from_pdf(file_path)
                    if text is not None:
                        output_file.write(text)
                        success_count += 1
                    else:
                        failure_count += 1
                elif file_path.lower().endswith('.txt'):
                    print(f"Reading text from {file_path}...")
                    text = extract_text_from_text_file(file_path)
                    if text is not None:
                        output_file.write(text)
                        success_count += 1
                    else:
                        failure_count += 1

    print(f"Text files merged successfully to '{output_file_path}'.")
    print(f"Total files processed: {success_count + failure_count}")
    print(f"Successful reads: {success_count}")
    print(f"Failed reads: {failure_count}")

folder_path = r'dataset'  # Use raw string for Windows paths
output_file_path = 'output_text_file.txt'
merge_text_files(folder_path, output_file_path)
