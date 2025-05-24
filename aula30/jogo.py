import numpy as np
import random
import os

WEIGHTS_FILE = 'perceptron_weights.npz'

class Perceptron:
    def __init__(self, dim, eta=0.1):
        self.w = np.zeros(dim)
        self.b = 0.0
        self.eta = eta
        self.load_weights()

    def predict(self, x):
        return 1 if (np.dot(self.w, x) + self.b) >= 0 else -1

    def train(self, x, t):
        y = self.predict(x)
        if y != t:
            self.w += self.eta * t * x
            self.b += self.eta * t

    def save_weights(self):
        np.savez(WEIGHTS_FILE, w=self.w, b=self.b)

    def load_weights(self):
        if os.path.exists(WEIGHTS_FILE):
            data = np.load(WEIGHTS_FILE)
            self.w = data['w']
            self.b = data['b']

class TicTacToe:
    def __init__(self):
        self.board = np.zeros(9, dtype=int)
        self.current = 1  # 1 = X, -1 = O

    def reset(self):
        self.board.fill(0)
        self.current = 1
        return self._get_state()

    def available_moves(self):
        return [i for i, v in enumerate(self.board) if v == 0]

    def step(self, action):
        if self.board[action] != 0:
            raise ValueError("Invalid move")
        self.board[action] = self.current
        winner = self._check_winner()
        done = winner is not None or not self.available_moves()
        reward = 0
        if done:
            if winner == 1:
                reward = 1
            elif winner == -1:
                reward = -1
        self.current *= -1
        return self._get_state(), reward, done

    def _get_state(self):
        return self.board.copy()

    def _check_winner(self):
        wins = [(0,1,2), (3,4,5), (6,7,8),
                (0,3,6), (1,4,7), (2,5,8),
                (0,4,8), (2,4,6)]
        for a,b,c in wins:
            s = self.board[a] + self.board[b] + self.board[c]
            if s == 3:
                return 1
            if s == -3:
                return -1
        return None


def train_perceptron(perceptron, episodes=10000):
    env = TicTacToe()
    for _ in range(episodes):
        state = env.reset()
        done = False
        states_actions = []
        while not done:
            if env.current == 1:
                # Perceptron move
                moves = env.available_moves()
                scores = []
                for m in moves:
                    s = state.copy()
                    s[m] = 1
                    scores.append((perceptron.w.dot(s) + perceptron.b, m))
                _, action = max(scores)
                states_actions.append((state.copy(), action))
            else:
                action = random.choice(env.available_moves())

            next_state, reward, done = env.step(action)
            state = next_state

        # assign reward for all X moves
        final_reward = reward  # +1, -1, or 0
        for st, act in states_actions:
            x = st.copy()
            x[act] = 1
            perceptron.train(x, np.sign(final_reward))

    perceptron.save_weights()
    print("Training completed and weights saved.")


def human_vs_perceptron(perceptron):
    env = TicTacToe()
    state = env.reset()
    print("Bem-vindo ao Jogo da Velha! Você é O (–1), máquina é X (+1).")
    done = False
    states_actions = []
    while not done:
        if env.current == 1:
            moves = env.available_moves()
            scores = []
            for m in moves:
                s = env.board.copy()
                s[m] = 1
                scores.append((perceptron.w.dot(s) + perceptron.b, m))
            _, action = max(scores)
            states_actions.append((env.board.copy(), action))
            print(f"Máquina joga em {action}")
        else:
            moves = env.available_moves()
            print(f"Posições disponíveis: {moves}")
            try:
                action = int(input("Sua jogada (0-8): "))
            except ValueError:
                print("Entrada inválida, use um número de 0 a 8.")
                continue
            if action not in moves:
                print("Jogada inválida!")
                continue

        state, reward, done = env.step(action)
        print_board(state)

        if done:
            winner = env._check_winner()
            if winner == 1:
                print("Máquina (X) venceu!")
            elif winner == -1:
                print("Você (O) venceu!")
            else:
                print("Empate!")

            # Aprendizado contínuo
            final_reward = reward  # +1, -1 ou 0
            for st, act in states_actions:
                x = st.copy()
                x[act] = 1
                perceptron.train(x, np.sign(final_reward))
            perceptron.save_weights()
            print("Pesos atualizados após a partida e salvos.")
            break


def print_board(state):
    symbols = {1: 'X', -1: 'O', 0: ' '}
    b = [symbols[v] for v in state]
    print(f"{b[0]}|{b[1]}|{b[2]}")
    print("-+-+-")
    print(f"{b[3]}|{b[4]}|{b[5]}")
    print("-+-+-")
    print(f"{b[6]}|{b[7]}|{b[8]}")

if __name__ == '__main__':
    percep = Perceptron(dim=9, eta=0.1)
    # Treinamento inicial se desejar
    if not os.path.exists(WEIGHTS_FILE):
        train_perceptron(percep, episodes=5000)
    human_vs_perceptron(percep)