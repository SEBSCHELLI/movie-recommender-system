import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

missing_values = ["", "\\N"]

# preprocess movies.csv
movies_df = pd.read_csv("data/raw/movies.csv", na_values=missing_values)

# get info about columns, non-null counts and dtypes
movies_df.info()

# remove rotten tomatoes columns rtID and rtPictureURL and imdbPictureURL column and spanish title column
movies_df = movies_df.drop(columns=["rtID", "rtPictureURL", "imdbPictureURL", "spanishTitle"])

# check those 4 movies missing the year specification
movies_df[movies_df["year"].isna()]

movies_df.loc[movies_df["year"].isna(), "year"] = 2007

# year column dtype is float64 -> can be changed to int
movies_df["year"].apply(float.is_integer).all()
movies_df["year"] = movies_df["year"].astype(np.int64)

# inspect duplicates where all id, title, imdbID and year are the same
movies_df[movies_df.duplicated(subset=('id', 'title', 'imdbID', 'year'), keep=False)]

# inspect duplicates where title, imdbID and year are the same
movies_df[movies_df.duplicated(subset=('title', 'imdbID', 'year'), keep=False)].sort_values(by=["title"])
# id column is not unique
# use imdbID as unique column now
ids_df = movies_df.groupby(['title', 'imdbID', 'year'])['id'].apply(list).reset_index(name='ids')['ids']
id_transform = {id:ids[0] for ids in ids_df.values for id in ids}
movies_df['id'] = movies_df['id'].map(id_transform)

movies_df = movies_df.drop_duplicates(subset=['id', 'title', 'imdbID', 'year'])

# check movies where title and year are the same
movies_df[movies_df.duplicated(subset=('title', 'year'), keep=False)].sort_values(by="title")
# rows with id 8484 and 26048 share the same title and year but are part 1 and part 2
# rows with id 4731 and 27728 are the same movie. Row with id 4731 has the correct imdbID
id_transform[27728] = 4731
movies_df = movies_df.drop([8362])


# check movies where title and imdbID are the same
movies_df[movies_df.duplicated(subset=('title', 'imdbID'), keep=False)].sort_values(by=['title'])


# check movies where imdbID and year are the same
movies_df[movies_df.duplicated(subset=('imdbID', 'year'), keep=False)]
# Title is different - > change id from 1 to 3114
id_transform[1] = 3114
movies_df = movies_df.drop([0])

movies_df.to_csv('data/preprocessed/movies.csv', index=False)




# preprocess ratings.csv
ratings_df = pd.read_csv("data/raw/ratings.csv", na_values=missing_values)

ratings_df.describe()
ratings_df.info()

ratings_df['movieID'] = ratings_df['movieID'].map(id_transform)


# check for duplicates after movieID transform
ratings_df[ratings_df.duplicated(subset=('user_id', 'movieID', 'rating'), keep=False)].sort_values(by=['user_id','movieID'])
ratings_df = ratings_df.drop_duplicates(subset=('user_id', 'movieID', 'rating'))

# check for duplicate rating from same user for same movie
ratings_df[ratings_df.duplicated(subset=('user_id', 'movieID'), keep=False)].sort_values(by=['user_id','movieID'])

# transform duplicate ratings
# 1. option: calculate mean rating
ratings_df = ratings_df.groupby(['user_id', 'movieID'])['rating'].mean().reset_index(name='rating')
ratings_df.to_csv('data/preprocessed/ratings.csv', index=False)



# create 5 folds for cv and store train and test folds as csv
cv = KFold(n_splits=5, shuffle=True)

num_users = ratings_df["user_id"].nunique()
num_movies = ratings_df["movieID"].nunique()

udict = {}
for i, u in enumerate(ratings_df["user_id"].unique()):
    udict[int(u)] = i

mdict = {}
for i, m in enumerate(ratings_df["movieID"].unique()):
    mdict[int(m)] = i

split_idxs = cv.split(ratings_df)
for i, (train_idxs, test_idxs) in enumerate(split_idxs):
    train = ratings_df.iloc[train_idxs]
    train.to_csv(f'data/cv/cv_train_{i}.csv', index=False)

    test = ratings_df.iloc[test_idxs]
    test.to_csv(f'data/cv/cv_test_{i}.csv', index=False)

    train_ratings = np.zeros((num_users, num_movies), dtype='float32')
    test_ratings = np.zeros((num_users, num_movies), dtype='float32')

    for row in train.itertuples():
        train_ratings[udict[int(row[1])], mdict[row[2]]] = int(row[3])

    for row in test.itertuples():
        test_ratings[udict[row[1]], mdict[row[2]]] = int(row[3])

    with open(f'data/cv/cv_train_{i}.npy', 'wb') as f:
        np.save(f, train_ratings)

    with open(f'data/cv/cv_test_{i}.npy', 'wb') as f:
        np.save(f, test_ratings)