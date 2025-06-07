import re

def clean_filename(filename, client_id):
    filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)
    return f"{client_id}_{filename}"
