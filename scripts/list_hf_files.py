from huggingface_hub import HfApi

def main():
    api = HfApi()
    files = api.list_repo_files('takala/financial_phrasebank')
    for f in files:
        print(f)

if __name__ == '__main__':
    main()
