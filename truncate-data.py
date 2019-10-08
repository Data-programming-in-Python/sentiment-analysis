import pandas as pd


def truncate_data():
    full_data = pd.read_csv('rotten_tomatoes_reviews.csv')
    truncated_data = full_data.head(2000)
    truncated_data.to_csv('truncated.csv', index=False)


if __name__ == '__main__':
    truncate_data()
