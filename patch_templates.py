import os
import re

template_dir = r"d:\5th_Sem\industrial_training\Revive\templates"

# Patterns to replace routing and static paths
replacements = [
    (r'http://127\.0\.0\.1:5500/templates/index\.html', r"{{ url_for('index') }}"),
    (r'http://127\.0\.0\.1:5500/templates/about-us\.html', r"{{ url_for('about_us') }}"),
    (r'http://127\.0\.0\.1:5500/templates/services\.html', r"{{ url_for('services') }}"),
    (r'http://127\.0\.0\.1:5500/templates/elements\.html', r"{{ url_for('elements') }}"),
    (r'http://127\.0\.0\.1:5500/templates/contact\.html', r"{{ url_for('contact') }}"),
    (r'href="index\.html"', r'href="{{ url_for(\'index\') }}"'),
    (r'href="about-us\.html"', r'href="{{ url_for(\'about_us\') }}"'),
    (r'href="services\.html"', r'href="{{ url_for(\'services\') }}"'),
    (r'href="elements\.html"', r'href="{{ url_for(\'elements\') }}"'),
    (r'href="contact\.html"', r'href="{{ url_for(\'contact\') }}"'),
    (r'href="index-icons\.html"', r'href="{{ url_for(\'index_icons\') }}"'),
]

# Regex for static files. We only look at href and src, starting with css/, js/, img/, fonts/
# E.g. src="img/core-img/logo.png" -> src="{{ url_for('static', filename='img/core-img/logo.png') }}"
static_pattern = re.compile(r'(href|src)="((css|js|img|fonts|scss)/[^"]+)"')

for filename in os.listdir(template_dir):
    if filename.endswith(".html"):
        filepath = os.path.join(template_dir, filename)
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        for old, new in replacements:
            content = re.sub(old, new, content)

        # Apply static files replacement
        # \1 corresponds to href or src
        # \2 corresponds to the path (e.g. css/style.css)
        content = static_pattern.sub(r'\1="{{ url_for(\'static\', filename=\'\2\') }}"', content)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

print("Patching complete.")
