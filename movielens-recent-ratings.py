import datasets
import pandas as pd
import re
import json
from sklearn.model_selection import train_test_split

_CITATION = """\
@InProceedings{huggingface:dataset,
title = {MovieLens Ratings},
author={Ismail Ashraq, James Briggs},
year={2022}
}
"""

_DESCRIPTION = """\
This dataset streams recent user ratings from the MovieLens 25M dataset and adds poster URLs.
"""
_HOMEPAGE = "https://grouplens.org/datasets/movielens/"

_LICENSE = ""

_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"

class MovieLens(datasets.GeneratorBasedBuilder):
    """The MovieLens 25M dataset for ratings"""

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "imdb_id": datasets.Value("string"),
                    "movie_id": datasets.Value("int32"),
                    "user_id": datasets.Value("int32"),
                    "rating": datasets.Value("float32"),
                    "title": datasets.Value("string"),
                    "poster": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage="https://grouplens.org/datasets/movielens/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        new_url = dl_manager.download_and_extract(_URL)
        # PREPROCESS
        # load all files
        movie_ids = pd.read_csv(new_url+"/ml-25m/links.csv")
        movie_meta = pd.read_csv(new_url+"/ml-25m/movies.csv")
        movie_ratings = pd.read_csv(new_url+"/ml-25m/ratings.csv")
        # merge to create movies dataframe
        movies = movie_meta.merge(movie_ids, on="movieId")
        # keep only subset of recent movies
        recent_movies = movies[movies["imdbId"].astype(int) >= 2000000].fillna("None")
        # mask movie ratings for movies that exist in movies
        mask = movie_ratings['movieId'].isin(recent_movies["movieId"])
        filtered_movie_ratings = movie_ratings[mask]
        # merge with movies
        df = filtered_movie_ratings.merge(
            recent_movies, on="movieId"
        ).astype(
            {"movieId": int, "userId": int, "rating": float}
        )
        # remove user and movies which occurs only once in the dataset
        df = df.groupby("movieId").filter(lambda x: len(x) > 2)
        df = df.groupby("userId").filter(lambda x: len(x) > 2)
        # convert unique movie IDs to sequential index values
        unique_movieids = sorted(df["movieId"].unique())
        mapping = {unique_movieids[i]: i for i in range(len(unique_movieids))}
        df["movie_id"] = df["movieId"].map(lambda x: mapping[x])
        # get unique user sequential index values
        unique_userids = sorted(df["userId"].unique())
        mapping = {unique_userids[i]: i for i in range(len(unique_userids))}
        df["user_id"] = df["userId"].map(lambda x: mapping[x])
        # add "tt" prefix to align with IMDB URL IDs
        df["imdb_id"] = df["imdbId"].apply(lambda x: "tt" + str(x))
        # now add the movie posters
        posters = datasets.load_dataset("pinecone/movie-posters", split='train').to_pandas()
        df = df.merge(posters, left_on='imdb_id', right_on='imdbId')
        # we also don't need all columns
        df = df[
            ["imdb_id", "movie_id", "user_id", "rating", "title", "poster"]
        ]
        # create train-test split
        train, test = train_test_split(
            df, test_size=0.1, shuffle=True, stratify=df["movie_id"], random_state=0
        )
        # save
        train.to_json(new_url+"/train.jsonl", orient="records", lines=True)
        test.to_json(new_url+"/test.jsonl", orient="records", lines=True)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": new_url+"/train.jsonl"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": new_url+"/test.jsonl"}
            ),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        with open(filepath, "r") as f:
            id_ = 0
            for line in f:
                yield id_, json.loads(line)
                id_ += 1
