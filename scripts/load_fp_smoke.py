from vibevalue.data import load_hf_dataset, validate_df

def main():
    print('Loading takala/financial_phrasebank sentences_50agree...')
    df = load_hf_dataset()
    print('Total rows:', len(df))
    print('Label distribution:')
    print(df['label'].value_counts().to_dict())
    print('\nFirst row:')
    print(df.iloc[0].to_dict())
    print('\nValidating...')
    dup = validate_df(df)
    print('Duplicate count:', dup)

if __name__ == '__main__':
    main()
