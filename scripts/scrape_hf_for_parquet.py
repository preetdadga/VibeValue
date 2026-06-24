import urllib.request
from html import unescape

base_page = 'https://huggingface.co/datasets/takala/financial_phrasebank'
print('Fetching', base_page)
html = urllib.request.urlopen(base_page, timeout=20).read().decode('utf-8', errors='ignore')
hits = set()
for part in html.split('href="'):
    if '.parquet' in part:
        url = part.split('"')[0]
        if url.startswith('/'):
            url = 'https://huggingface.co' + url
        hits.add(unescape(url))

print('Found parquet links:')
for h in sorted(hits):
    print(h)
