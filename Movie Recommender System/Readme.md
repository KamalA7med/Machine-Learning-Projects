# Movie Recommenders in PyTorch
 
Two movie recommendation systems built from scratch in PyTorch on the MovieLens dataset:
 
1. **Collaborative Filtering** with Matrix Factorization
2. **Content-Based Filtering** with a Two-Tower Neural Network
The notebook covers data preprocessing, model definitions, custom training loops, evaluation, and generating recommendations for a brand-new user.
 
## Dataset
 
[MovieLens](https://grouplens.org/datasets/movielens/) small dataset (610 users, 9,742 movies). The files used are `movies.csv`, `ratings.csv`, `tags.csv` and `links.csv`, placed in a folder named `Movies lens/`.
 
Preprocessing:
 
- Keep only movies with at least 10 ratings, which leaves 2,269 movies.
- Pivot the ratings into a movie x user matrix (2,269 x 610) and fill missing values with 0.
- Randomly split the known ratings into train and test sets (about 80/20).
## Approach 1: Matrix Factorization (collaborative filtering)
 
Each user and each movie gets a learned vector of 25 latent features. The predicted rating is the dot product of the two vectors plus a user bias.
 
- **Parameters:** user matrix `W`, user bias `b`, movie matrix `X`
- **Loss:** squared error over known ratings only, plus L2 regularization (lambda = 20)
- **Optimizer:** Adam, learning rate 0.1, 200 epochs
**What you can do with it:**
 
- **Similar movies:** cosine similarity between the learned movie vectors. For example, the closest movies to *Toy Story* include *Aladdin* and *Mulan*.
- **Recommendations for a new user:** the movie matrix is frozen and the new user's vector is found with a closed-form ridge regression, so recommendations come back instantly without retraining.
## Approach 2: Two-Tower Neural Network (content-based filtering)
 
Two separate networks map a user and a movie into a shared 32-dimensional space. The predicted rating is the dot product of the two outputs.
 
- **Movie features:** multi-hot genre vector (19 genres)
- **User features:** the user's average rating per genre (scaled to 0-1), computed from training ratings only to avoid data leakage
- **Each tower:** 19 -> 256 -> 128 -> 64 -> 32 with ReLU and dropout
- **Loss:** MSE
- **Optimizer:** Adam, learning rate 0.001, 2,000 epochs (full batch)
To recommend for a new user, their ratings are turned into a genre profile, passed through the user tower, and scored against every movie from the movie tower.
 
## Results
 
| Model | Train RMSE | Test RMSE |
|-------|-----------|-----------|
| Matrix Factorization | 0.81 | 0.86 |
| Two-Tower Network | 0.83 | 0.88 |
 
Both models predict ratings to within about 0.9 stars on unseen ratings. Matrix factorization is slightly more accurate, while the two-tower model works from genre information only, so it can score movies it has little rating data for.
 
## Requirements
 
- Python 3.8+
- PyTorch
- pandas
- numpy
- scikit-learn
- matplotlib
```bash
pip install torch pandas numpy scikit-learn matplotlib
```
 
A GPU speeds up training but is not required. The notebook was run on a Google Colab T4 GPU.
 
## How to run
 
1. Download the MovieLens small dataset and put the CSV files in a folder named `Movies lens/` next to the notebook.
2. Open `Building_Movie_Recommenders_in_PyTorch_Collaborative_and_Content_Based_Filtering.ipynb` in Jupyter or Google Colab.
3. Run all cells from top to bottom.
4. Edit the `my_ratings` dictionary near the end to get recommendations for your own taste.
## Project structure
 
```
.
├── Building_Movie_Recommenders_in_PyTorch_Collaborative_and_Content_Based_Filtering.ipynb
├── Movies lens/
│   ├── movies.csv
│   ├── ratings.csv
│   ├── tags.csv
│   └── links.csv
└── README.md
```
 
## Limitations and ideas
 
- **Cold start:** new users need some ratings before recommendations are meaningful.
- **Genres only:** the two-tower model ignores tags, release year and other metadata. Adding the tags and year as features is a natural next step.
- **Random split:** a time-based split would better reflect real-world use.
- **Metrics:** adding ranking metrics such as precision@k and recall@k would show how good the top-10 lists are, which matters more than RMSE for recommendations.
## What I learned
 
- How matrix factorization learns latent user and movie features from a sparse rating matrix.
- How to mask missing ratings in the loss so only known ratings count.
- How to build a two-tower network and train it with a dot-product output.
- How to produce recommendations for a new user without retraining the model.
 
