import os
import random
import torch
from tic_tac_toe_agent_training import net, choose_action, Legal_Moves, Check_Winner


def show_board(board):
    cells = []
    for i in range(9):
        if board[i] == 1:
            cells.append("X")
        elif board[i] == -1:
            cells.append("O")
        else:
            cells.append(" ")          # empty square: show nothing
    print()
    print(f" {cells[0]} | {cells[1]} | {cells[2]}")
    print("---+---+---")
    print(f" {cells[3]} | {cells[4]} | {cells[5]}")
    print("---+---+---")
    print(f" {cells[6]} | {cells[7]} | {cells[8]}")
    print()

def ask_human_move(board):
    while True:
        text = input("Your move (0-8): ")
        if text.isdigit() and int(text) in Legal_Moves(board):
            return int(text)
        print("Not allowed. Pick a free square number.")


def play_game():
    board = [0] * 9                      # agent = X (1), you = O (-1)

    if random.random() < 0.5:            # sometimes the agent starts
        print("Agent starts.")
        board[choose_action(board, epsilon=0)] = 1
    else:
        print("You start.")

    while True:
        show_board(board)
        board[ask_human_move(board)] = -1           # your move
        winner = Check_Winner(board)
        if winner is not None:
            break

        board[choose_action(board, epsilon=0)] = 1  # agent's move
        winner = Check_Winner(board)
        if winner is not None:
            break

    show_board(board)
    if winner == -1:
        print("You won!")
    elif winner == 1:
        print("The agent won.")
    else:
        print("Draw.")


if __name__ == "__main__":
    if not os.path.exists("xo_agent.pth"):
        print("No saved agent found. Run your training file first.")
    else:
        net.load_state_dict(torch.load("xo_agent.pth"))
        print("Loaded the trained agent. You are O, the agent is X.")
        while True:
            play_game()
            if input("Play again? (y/n): ").lower() != "y":
                break
