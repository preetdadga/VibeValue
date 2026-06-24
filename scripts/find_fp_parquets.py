import urllib.request

base = 'https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/'
candidates = [
    'sentences_50agree/financial_phrasebank.parquet',
    'sentences_50agree/train.parquet',
    'sentences_50agree.parquet',
    'parquet/sentences_50agree.parquet',
    'sentences_50agree/data.parquet',
    'all.parquet',
]

found = []
for c in candidates:
    url = base + c
    try:
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req, timeout=10) as r:
            print('OK:', url, r.status)
            found.append(url)
    except Exception as e:
        print('MISS:', url, e)

print('\nFound:', found)
