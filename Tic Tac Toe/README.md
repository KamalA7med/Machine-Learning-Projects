# Tic-Tac-Toe (XO) DQN Agent

A Tic-Tac-Toe agent trained with **Deep Q-Learning (DQN)** in **PyTorch**. The agent learns by playing thousands of games against an opponent that mixes random moves with perfect minimax moves. The goal: **never lose, and win whenever the opponent makes a mistake.**

## How it works

Instead of a Q-table, a small neural network takes the board (9 squares) and outputs 9 scores, one per square. The highest-scoring free square is the move the agent plays.

The agent learns by:

1. Playing a move (sometimes random, to explore).
2. Getting a reward from the game result.
3. Moving the score of that move closer to `reward + 0.95 × (best score of the next position)`.

### Rewards

| Result | Reward |
|--------|--------|
| Win | +1.0 |
| Draw | +0.2 |
| Loss | -1.0 |
| Game continues | 0.0 |

### The opponent

The opponent plays as **O** and mixes two styles:

- **Random moves** make mistakes, so the agent learns to spot openings and win.
- **Minimax moves** are perfect, so the agent learns never to lose.

Minimax is only the training partner. The DQN agent never searches ahead; it learns a feel for good moves from experience.

### DQN ingredients

- **Replay memory:** stores past moves and learns from random samples, so it doesn't overfit to the last game.
- **Target network:** a slowly updated copy of the network that keeps training stable.
- **Epsilon-greedy exploration:** starts fully random (`1.0`) and decays to `0.05`.
- **Illegal move masking:** squares that are already taken can never be chosen.

## Project structure

```
.
├── tic_tac_toe_agent_training.py   # game rules, minimax opponent, DQN, training and evaluation
├── play.py                         # play against the trained agent in the terminal
├── xo_agent.pth                    # saved trained weights (created after training)
└── README.md
```

## Requirements

- Python 3.8+
- PyTorch

```bash
pip install torch
```

## Usage

### 1. Train the agent

```bash
python tic_tac_toe_agent_training.py
```

Training prints progress every 3,000 episodes, then evaluates the agent and saves the weights to `xo_agent.pth`.

### 2. Play against it

```bash
python play.py
```

You play as **O** and the agent plays as **X**. Type a square number from 0 to 8 to make your move:

```
 0 | 1 | 2
-----------
 3 | 4 | 5
-----------
 6 | 7 | 8
```

Who starts is random in every game.

> **Note:** run `play.py` from a real terminal. Output-only panels (for example the VS Code Output panel) can't accept keyboard input.

## Training settings

| Setting | Value |
|---------|-------|
| Episodes | 15,000 |
| Network | 9 → 128 → 128 → 9 (ReLU) |
| Optimizer | Adam, learning rate 0.001 |
| Discount factor (gamma) | 0.95 |
| Batch size | 64 |
| Replay memory size | 50,000 |
| Target network update | every 200 steps |
| Epsilon | 1.0 → 0.05 |
| Training opponent | 50% random moves, 50% perfect moves |

## Results

Evaluated over 1,000 games with no exploration:

| Opponent | Wins | Draws | Losses |
|----------|------|-------|--------|
| Fully random | 938 | 61 | 1 |
| Perfect (minimax) | 0 | 997 | 3 |

Against a perfect player, a draw is the best possible result, so the agent is very close to unbeatable. Exact numbers change from run to run.

## Tips

- DQN gives no mathematical guarantee of perfect play. If you see losses against the perfect opponent, train for more episodes (30,000 or more) or vary the opponent's randomness during training.
- To retrain from scratch, run the training script again. It overwrites `xo_agent.pth`.

## What I learned

- How Q-learning works and how a neural network replaces the Q-table.
- Why replay memory and a target network make training stable.
- How minimax finds perfect moves by searching every possible future.
