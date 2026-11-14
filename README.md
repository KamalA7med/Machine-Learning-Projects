# Machine Learning Projects
 
A collection of machine learning and deep learning projects, from classic prediction tasks to recommender systems and reinforcement learning. Each project lives in its own folder with its code and notes.
 
## Projects
 
| Project | Type | What it does |
|---------|------|--------------|
| [Tic Tac Toe](./Tic%20Tac%20Toe) | Reinforcement learning (DQN) | An agent that learns to play Tic-Tac-Toe and almost never loses |
| [Movie Recommender System](./Movie%20Recommender%20System) | Recommender systems (deep learning) | Movie recommendations with matrix factorization and a two-tower neural network |
| [Titanic Survival Prediction](./Titanic_Survival_Prediction) | Classification | Predicts which passengers survived the Titanic |
| [Loan Approval Prediction](./Loan-Approval-Prediction) | Classification | Predicts whether a loan application will be approved |
| [Credit Card Fraud Detection](./Credit%20Card%20Fraud%20Detection) | Classification | Detects fraudulent credit card transactions |
| [Big Mart Sales Prediction](./Big_Mart_Sales_Prediction) | Regression | Predicts product sales for Big Mart outlets |
 
## Featured projects
 
### Tic Tac Toe: Deep Q-Learning agent (PyTorch)
 
A neural network learns to play Tic-Tac-Toe by playing thousands of games against an opponent that mixes random moves with perfect minimax moves. Training uses replay memory, a target network, epsilon-greedy exploration and illegal-move masking.
 
Result over 1,000 games: 938 wins, 61 draws and 1 loss against a random player, and 997 draws and 3 losses against a perfect player (a draw is the best possible result).
 
### Movie Recommender System (PyTorch)
 
Two recommenders built from scratch on the MovieLens dataset:
 
- **Collaborative filtering** with matrix factorization (25 latent features), plus a similar-movies search and instant recommendations for a new user.
- **Content-based filtering** with a two-tower neural network that learns from movie genres and each user's genre preferences.
Test RMSE: 0.86 for matrix factorization and 0.88 for the two-tower network.
 
## Tech stack
 
- Python
- PyTorch
- scikit-learn
- pandas and NumPy
- Matplotlib
- Jupyter Notebook and Google Colab
## Getting started
 
```bash
git clone https://github.com/KamalA7med/<your-repo-name>.git
cd <your-repo-name>
pip install torch pandas numpy scikit-learn matplotlib
```
 
Then open a project folder and follow the instructions inside it (or open its notebook and run the cells from top to bottom). Some projects need a dataset downloaded first. Check the project folder for details.
 
## Repository structure
 
```
.
├── Big_Mart_Sales_Prediction/
├── Credit Card Fraud Detection/
├── Loan-Approval-Prediction/
├── Movie Recommender System/
├── Tic Tac Toe/
├── Titanic_Survival_Prediction/
└── README.md
```
 
## About
 
I'm learning and practicing machine learning by building end-to-end projects: cleaning data, training models, evaluating them honestly and explaining the results.
 
Find me on GitHub: [KamalA7med](https://github.com/KamalA7med)
 
