import pandas as pd

def get_words_set(df): return {word for index, row in df.iterrows() for word in row['Review'].split()}

def find_irrelevant_words():

    cleaned = pd.read_csv('cleaned.csv')

    word_set = get_words_set(cleaned)

    print('word_set size: ', len(word_set))

    word_count = {1: {}, 0: {}}

    for index, row in cleaned.iterrows():
        freshness = row[0]
        review = row[1]

        words = review.split()

        for word in words:
            if word in word_count[freshness].keys():
                word_count[freshness][word] += 1
            else:
                word_count[freshness][word] = 1

    highest_fresh = ''
    highest_fresh_count = 0
    highest_rotten = ''
    highest_rotten_count = 0

    for freshness in word_count.keys():
        for word in word_count[freshness].keys():
            if freshness == 0:
                if word_count[freshness][word] > highest_rotten_count:
                    print(freshness, word, word_count[freshness][word])
                    highest_rotten_count = word_count[freshness][word]
                    highest_rotten = word
            if freshness == 1:
                if word_count[freshness][word] > highest_fresh_count:
                    print(freshness, word, word_count[freshness][word])
                    highest_fresh_count = word_count[freshness][word]
                    highest_fresh = word

    print('number of fresh records: ', len(word_count[1].keys()))
    print('number of rotten records: ', len(word_count[0].keys()))

    print('highest_rotten:', highest_rotten)
    print('highest_rotten_count:', highest_rotten_count)
    print('highest_fresh:', highest_fresh)
    print('highest_fresh_count:', highest_fresh_count)

    word_ratio_report = open('word_ratio_report.csv', 'w')

    word_ratio_report.write('Word,Fresh Count,Rotten Count,Ratio(Fresh to Rotten),Absolute Difference\n')
    for word in word_set:
        fresh_count = 0
        rotten_count = 0
        if word in word_count[1].keys():
            fresh_count = word_count[1][word]
        if word in word_count[0].keys():
            rotten_count = word_count[0][word]

        try:
            ratio = fresh_count / rotten_count
        except ZeroDivisionError:
            ratio = 25
        absolute_differences = abs(fresh_count - rotten_count)

        word_ratio_report.write(f'{word},{fresh_count},{rotten_count},{ratio},{absolute_differences}\n')

    word_ratio_report.close()



if __name__ == '__main__':
    find_irrelevant_words()