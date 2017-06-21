import pandas as pd
import numpy as np
from textblob import TextBlob
import os
import csv
from sklearn.neighbors import KNeighborsClassifier
import json
from IPython import embed


def updateDictionary(score, in_id, in_dict):
    if in_id not in in_dict:
        in_dict[in_id] = (score, 1)
    else:
        prev_mean = in_dict[in_id][0]
        prev_count = in_dict[in_id][1]
        new_count = prev_count + 1
        new_mean = (prev_mean * (prev_count/new_count)) + (score * (1/new_count))
        in_dict[in_id] = (new_mean, new_count)


def makeProductandUserDictionary():
    print("making test labels...")
    product_id_dict = {}
    user_id_dict = {}
    with open("./data/all_data/train.csv", "rt", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=",")
        iterreader = iter(reader)
        next(iterreader)
        for i, row in enumerate(iterreader):
            if i % 1000 == 0:
                print(i)
            product_id = row[3]
            user_id = row[8]
            score = int(row[4])

            updateDictionary(score, product_id, product_id_dict)
            updateDictionary(score, user_id, user_id_dict)

    with open('full_product_id_dict.json', 'w') as fp:
        json.dump(product_id_dict, fp)

    with open('full_user_id_dict.json', 'w') as fp:
        json.dump(user_id_dict, fp)


def makeLabelsCSV(prefix, fn, length):
    labels = pd.DataFrame(index=np.arange(0, length), columns=['Score'])
    with open(prefix + fn, "rt", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=",")
        iterreader = iter(reader)
        next(iterreader)
        for i, row in enumerate(iterreader):
            if i % 1000 == 0:
                print(i)
            score = int(row[4])
            labels.loc[i] = [score]
    if fn == 'train.csv':
        labels.to_csv('train_labels.csv', sep=',', index=False, encoding='utf-8')
    else:
        labels.to_csv('test_labels.csv', sep=',', index=False, encoding='utf-8')


def preprocess(ToT, prefix, set_len):
    print("preprocessing..." + ToT)

    # Open the dictionary
    with open('full_product_id_dict.json', 'r') as fp:
        product_id_dict = json.load(fp)

    with open('full_user_id_dict.json', 'r') as fp:
        user_id_dict = json.load(fp)

    if ToT == "train":
    # Build a dataframe with all the data
        processed_data = pd.DataFrame(index=np.arange(0, set_len), columns=('ReviewId', 'ProductId', 'HelpfulnessFraction', 'SummaryPolarity', 'SummarySubjectivity', 'TextPolarity', 'ProductAverage', 'UserAverage'))
    else:
        processed_data = pd.DataFrame(index=np.arange(0, set_len), columns=('ReviewId', 'ProductId', 'HelpfulnessFraction', 'SummaryPolarity', 'SummarySubjectivity', 'TextPolarity', 'ProductAverage', 'UserAverage'))

    with open(prefix + ToT + ".csv", "rt", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=",")
        iterreader = iter(reader)
        next(iterreader)
        for i, row in enumerate(iterreader):
            #           [ 0                         1                       2     3            4        5          6       7       8]
            # line[0] = ['HelpfulnessDenominator', 'HelpfulnessNumerator', 'Id', 'ProductId', 'Score', 'Summary', 'Text', 'Time', 'UserId']
            if i % 1000 == 0:
                print(i)
            # Get ProductId from Dict
            product_id = row[3]
            user_id = row[8]
            # Calculate the helpfulness fraction
            try:
                # Helpfulness Calculation
                old_helpfulness_fraction = .75  # average
                if float(row[0]):
                    old_helpfulness_fraction = float(row[1])/float(row[0])
                    helpfulness_fraction = ((old_helpfulness_fraction) * (2)) - 1
                else:
                    helpfulness_fraction = ((old_helpfulness_fraction) * (2)) - 1
                # Product Avg Calculation
                # NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
                if product_id in product_id_dict:
                    if product_id_dict[product_id][1]:
                        old_avg = product_id_dict[product_id][0]
                        product_avg = (((old_avg) * (2)) / (5)) - 1
                else:
                    old_avg = 4.2
                    product_avg = (((old_avg) * (2)) / (5)) - 1

                if user_id in user_id_dict:
                    if user_id_dict[user_id][1]:
                        old_avg = user_id_dict[user_id][0]
                        user_avg = (((old_avg) * (2)) / (5)) - 1
                else:
                    old_avg = 4.2
                    user_avg = (((old_avg) * (2)) / (5)) - 1

                # Sentiment Analysis on Summary and Text
                summary = TextBlob(row[5])
                text = TextBlob(row[6])
                # Store the calculations in a dataframe
                processed_data.loc[i] = [
                    row[2],  # ReviewID
                    product_id,  # ProductId
                    helpfulness_fraction,  # HelpfulnessFraction [-1, 1]
                    summary.sentiment.polarity,  # SummaryPolarity [-1, 1]
                    summary.sentiment.subjectivity,  # SummarySubjectivity [-1, 1]
                    text.sentiment.polarity,  # TextPolarity [-1, 1]
                    product_avg,  # ProductAverage [-1, 1]
                    user_avg  # UserAverage [-1, 1]
                    ]
            except:
                print(row)
        # Save dataframe to csv
        processed_data.to_csv(ToT + '_processed_data.csv', sep=',', index=False, encoding='utf-8')


def createMask(train_labels):
    score_count = [0] * 5
    msk = [False] * len(train_labels.index)
    num_true = 0
    max_each = 40000
    for i in range(len(train_labels.index)):
        score_idx = train_labels.Score[i] - 1
        if score_count[score_idx] < max_each:
            num_true += 1
            msk[i] = True
        if num_true == max_each * 5:
            break
    return msk


def train(num_neighbors):
    print("training...")
    clean_train_data = pd.read_csv("train_processed_data.csv", quotechar='"', skipinitialspace=True, index_col=False)
    clean_train_labels = pd.read_csv("train_labels.csv", quotechar='"', skipinitialspace=True, index_col=False)
    clean_train_data.drop(['ReviewId', 'ProductId'], axis=1, inplace=True)

    # APPLY MASK TO BALANCE DATA
    msk = createMask(clean_train_labels)
    balanced_train_data = clean_train_data[msk]
    balanced_train_labels = clean_train_labels[msk]
    neigh = KNeighborsClassifier(n_neighbors=num_neighbors, weights='uniform', algorithm='auto', leaf_size=10, metric='manhattan')
    neigh.fit(balanced_train_data, balanced_train_labels.values.ravel())
    return neigh, balanced_train_labels


def test(size, max_size, num_neighbors, test_len, neigh, balanced_train_labels):
    print("testing...")
    clean_test_data = pd.read_csv("test_processed_data.csv", quotechar='"', skipinitialspace=True, index_col=False)
    review_ids = clean_test_data.ReviewId
    clean_test_data.drop(['ReviewId', 'ProductId'], axis=1, inplace=True)
    dist, all_neighbors = neigh.kneighbors(clean_test_data, num_neighbors)
    predicted_labels = pd.DataFrame(index=np.arange(0, test_len), columns=('Id', 'Score'))

    predictions = [0] * num_neighbors
    for review_idx, neighbors in enumerate(all_neighbors):
        if review_idx % 1000 == 0:
            print(review_idx)
        for local_idx, global_idx in enumerate(neighbors):
            predictions[local_idx] = balanced_train_labels.Score[global_idx]
        predicted_label = np.mean(predictions)

        review_id = review_ids[review_idx]
        predicted_labels.loc[review_idx] = [review_id, predicted_label]
    predicted_labels.to_csv('submission.csv', sep=',', index=False, encoding='utf-8')
    if size < max_size:
        test_labels = pd.read_csv("test_labels.csv", quotechar='"', skipinitialspace=True, index_col=False)
        rmse = ((predicted_labels.Score - test_labels.Score) ** 2).mean() ** .5
        print(rmse)


def main():
    size = 5

    if size == 1:
        prefix = './data/small_data/'
        test_len = 9
        train_len = 9999
    elif size == 2:
        prefix = './data/medium_data/'
        test_len = 9999
        train_len = 9999
    elif size == 3:
        prefix = './data/balanced_data/'
        test_len = 9999
        train_len = 9999
    elif size == 4:
        prefix = './data/big_data/'
        test_len = 9999
        train_len = 49999
    elif size == 5:
        prefix = './data/100_train_10_test_data/'
        test_len = 9999
        train_len = 99999
    elif size == 6:  # DOES NOT WORK
        prefix = './data/helpfulness_data/'
        test_len = 9999
        train_len = 49999
    elif size == 7:
        prefix = './data/200_train/'
        test_len = 100000
        train_len = 199999

    max_size = 7
    # PREPROCESS
    if not (os.path.isfile("full_product_id_dict.json") and os.path.isfile("full_user_id_dict.json")):
        makeProductandUserDictionary()

    if not (os.path.isfile("train_labels.csv")):
        print("making train_labels...")
        makeLabelsCSV(prefix, 'train.csv', train_len)

    if not (os.path.isfile("test_labels.csv")):
        print("making test_labels...")
        makeLabelsCSV(prefix, 'test.csv', test_len)

    if not (os.path.isfile("train_processed_data.csv")):
        preprocess("train", prefix, train_len)

    if not (os.path.isfile("test_processed_data.csv")):
        preprocess("test", prefix, test_len)

    # TRAIN
    num_neighbors = 25
    neigh, balanced_train_labels = train(num_neighbors)

    # TEST
    test(size, max_size, num_neighbors, test_len, neigh, balanced_train_labels)

if __name__ == '__main__':
    np.random.seed(3)
    main()
