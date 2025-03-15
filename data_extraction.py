import pandas as pd

df = pd.read_csv('shortjokes.csv')

if 'Joke' in df.columns:
    jokes = df['Joke'].head(500)

    with open('top_100_jokes.txt', 'w', encoding='utf-8') as f:
        for joke in jokes:
            f.write(joke + '\n\n') 

    print("Top 100 jokes saved to 'top_100_jokes.txt'")
else:
    print("Column 'Jokes' not found in the CSV file.")
