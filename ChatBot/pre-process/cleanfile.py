import re

def clean_text(text):
    # Remove non-printable characters
    cleaned_text = ''.join(char for char in text if char.isprintable())

    # Remove special characters and symbols other that (, . ! ? : / - ' " ;)
    cleaned_text = re.sub(r'[^\w\s.,!?/:;"-]', '', cleaned_text)

    # Remove extra whitespace at the beginning and end of the text
    cleaned_text = cleaned_text.strip()

    return cleaned_text

# Read the input text file
input_file_path = 'output_text_file.txt' 

try:
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        text_content = input_file.read()
except Exception as e:
    print(f"Error reading input file: {e}")
    exit(1)

# Clean the text content
cleaned_text = clean_text(text_content)

# Write the cleaned text to a new file
output_file_path = 'cleaned_output_text_file.txt'  

try:
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(cleaned_text)
except Exception as e:
    print(f"Error writing output file: {e}")
    exit(1)

print(f"Cleaned text saved to '{output_file_path}'.")
