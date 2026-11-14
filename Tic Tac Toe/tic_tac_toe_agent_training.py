import numpy as np
import torch.nn as nn
import  random
import torch
from collections import deque
Win_Lines=[(0,1,2),(3,4,5),(6,7,8),(0,4,8),(2,4,6),(0,3,6),(1,4,7),(2,5,8)]
def Check_Winner(board):
     for a,b,c  in Win_Lines:
         if board[a]!=0 and board[a]==board[b]==board[c]:
             return board[a]
     if 0 not in board:
         return 0    
     return None
 
def Legal_Moves(board):
    return [i for i in range(9)if board[i]==0]  


cache={}
def MINIMAX(board,player):
    key=(tuple(board),player)
    if key in cache:
        return cache[key]
    Winner=Check_Winner(board)
    if Winner is not None:
        if Winner==player:
            score=1
        elif Winner==0:
            score=0
        else :
            score=-1        
            
    else :
         best_score=-2
         for move in Legal_Moves(board):
            board[move]=player
            score=-MINIMAX(board,-player)
            board[move]=0
            best_score=max(best_score,score)
         score=best_score
    cache[key]=score
    return score


def perfect_move(board,player):
    best_moves=[]
    best_score=-2
    for move in Legal_Moves(board):
        board[move]=player
        score=-MINIMAX(board,-player)
        board[move]=0
        if score>best_score:
            best_score=score
            best_moves=[move]
        elif score==best_score:
            best_moves.append(move)    
        
    return random.choice(best_moves)

def oppenent_move(board,p):
    if random.random()<p:
        return random.choice(Legal_Moves(board))
    return perfect_move(board,-1)




class TicTacToe:
    def __init__(self,p_random):
        self.board=[0]*9
        self.p_random=p_random
    
    def reset(self):
        self.board=[0]*9
        if random.random()<0.5:
            move=oppenent_move(self.board,self.p_random) 
            self.board[move]=-1
        return list(self.board)
    
    def step(self,action):
        self.board[action]=1
        Winner=Check_Winner(self.board)
        if Winner is  None:
            move=oppenent_move(self.board,self.p_random)
            self.board[move]=-1
            Winner=Check_Winner(self.board)
        if Winner is None:
            return list(self.board),0,False
        elif Winner==0:
            return list(self.board),0.2,True
        elif Winner==1:
            return list(self.board),1,True
        else:
            return list(self.board),-1,True


class make_network(nn.Module):
    def __init__(self):
        super().__init__()
        self.Layer1=nn.Linear(9,128)
        self.Layer2=nn.Linear(128,128)
        self.Layer3=nn.Linear(128,9)
    
    
    def forward(self,x):
        x=self.Layer1(x)
        x=torch.relu(x)
        x=self.Layer2(x)
        x=torch.relu(x)
        x=self.Layer3(x)
        return x
    
net=make_network()
target_net=make_network()
target_net.load_state_dict(net.state_dict())
optim=torch.optim.Adam(net.parameters(),lr=1e-3)

def choose_action(state,epsilon):
    if random.random()<epsilon:
        return random.choice(Legal_Moves(state))
    with torch.no_grad():
        scores=net(torch.tensor(state,dtype=torch.float32))
        for i in range(9):
            if state[i]!=0:
                scores[i]=-1e9
    return int(scores.argmax())

        
memory=deque(maxlen=50_000)
Gamma=0.95


def Learn(batch_size=64):
    if len(memory)<batch_size:
        return
    batch=random.sample(memory,batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    states = torch.tensor(states, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)    
    current_q = net(states).gather(1, actions).squeeze(1)
    with torch.no_grad():
        next_scores=target_net(next_states)
        next_scores[next_states!=0]=-1e9     
        best_next=next_scores.max(1).values
        target_q=rewards+Gamma*best_next*(1-dones)
    
    optim.zero_grad()
    Loss= nn.functional.smooth_l1_loss(current_q, target_q) 
    Loss.backward()
    optim.step()




def train(episodes=15_000):
    env=TicTacToe(0.5)
    steps=0
    for episode in range(episodes):
        epsilon = max(0.05, 1.0 - episode / (episodes * 0.6))
        state=env.reset()
        done=False
        while not done:
            action=choose_action(state,epsilon)
            next_state,reward,done=env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            Learn()
            steps += 1
            if steps % 200 == 0:            
                target_net.load_state_dict(net.state_dict())

        if (episode + 1) % 3000 == 0:
            print("finished episode", episode + 1)
def evaluate(p_random, games=1000):
    env = TicTacToe(p_random)
    wins = draws = losses = 0

    for _ in range(games):
        state = env.reset()
        done = False
        while not done:
            action = choose_action(state, epsilon=0)
            state, reward, done = env.step(action)
        if reward == 1.0:
            wins += 1
        elif reward == -1.0:
            losses += 1
        else:
            draws += 1

    print(f"opponent random-prob={p_random}: wins={wins} draws={draws} losses={losses}")


if __name__ == "__main__":
    train()
    evaluate(1.0)  
    evaluate(0.0) 
    torch.save(net.state_dict(), "xo_agent.pth")
            
            
            


   

