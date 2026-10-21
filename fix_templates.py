import os
import re

template_dir = r"d:\5th_Sem\industrial_training\Revive\templates"

for filename in os.listdir(template_dir):
    if filename.endswith(".html"):
        filepath = os.path.join(template_dir, filename)
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Fix backslashes in url_for
        content = content.replace(r"url_for(\'", "url_for('")
        content = content.replace(r"\')", "')")
        content = content.replace(r"filename=\'", "filename='")

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

print("Fixed templates.")
