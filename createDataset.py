import pandas as pd
import re

if __name__ == "__main__":
    print("Data loading...")
    links = pd.read_csv("./MLdataset/links.csv")
    movies = pd.read_csv("./MLdataset/movies.csv")
    ratings = pd.read_csv("./MLdataset/ratings.csv")
    print("Data loaded.")

    movies_ratings_merge = movies.merge(ratings, left_on="movieId", right_on="movieId")
    df = movies_ratings_merge.merge(links, left_on="movieId", right_on="movieId")
    # print(f"Total length: {len(df)}")
    
    # filter movies with lower year bound creation
    YEAR_LOWER_BOUND = 1998
    YEAR_UPPER_BOUND = 2016
    df["year"] = df["title"].str.extract(r'\((\d{4})\)$').astype(float)
    df = df[(df["year"] >= YEAR_LOWER_BOUND) & (df["year"] <= YEAR_UPPER_BOUND)]

    # Filter by number of ratings
    MIN_RATINGS = 20
    movie_counts = df.groupby("movieId").size()
    filtered_movies = movie_counts[movie_counts > MIN_RATINGS].index
    df = df[df["movieId"].isin(filtered_movies)]

    # Filter users to each some minimum amount of ratings
    USER_MIN_RATINGS = 10
    user_counts = df.groupby("userId").size()
    filtered_users = user_counts[user_counts > USER_MIN_RATINGS].index
    df = df[df["userId"].isin(filtered_users)]

    print(f"Number of reviews after filtering: {len(df)}")

    if len(df) > 1_000_000:
        df = df.sample(n=1_000_000, random_state=42)
    
    df["user_id"], _ = pd.factorize(df["userId"])
    df["movie_id"], _ = pd.factorize(df["movieId"])

    # Save map new_movie_id -> info about that movie
    movie_lookup = df[["movie_id", "title", "imdbId", "tmdbId", "genres"]].drop_duplicates()
    movie_lookup.to_csv("movie_lookup.csv", index=False)

    df[["user_id", "movie_id", "rating"]].to_csv("movies_dataset.csv", index=False)

    print("Dataset created successfully.")
    
