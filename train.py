from agent import Agent
from SnakeGame import SnakeGame
import matplotlib.pyplot as plt

# Optional: for plotting scores
def plot(scores, mean_scores):
    '''
    This is for visualizing progress; 
    plotting the game score and average scores during training. 
    '''
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Game')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
    plt.pause(0.1)

def train():
    '''
    - Plays the game on loop (repeatedly).
    - Agent decides moves based on experience which will help improve over time. 
    - Scores progress as training continues.
    '''
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    agent = Agent()
    game = SnakeGame()

    while True:
        # 1. Get current state
        state_old = agent.get_state(game)

        # 2. Get action based on current state
        final_move, action_index = agent.get_action(state_old)

        # 3. Perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # 4. Train short memory
        agent.train_short_memory(state_old, action_index, reward, state_new, done)

        # 5. Store experience in memory
        agent.remember(state_old, action_index, reward, state_new, done)

        if done:
            # Game over - reset and train long memory
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            # Track high score
            if score > record:
                record = score
                agent.model.save()

            print(f'Game {agent.n_games} - Score: {score} - Record: {record}')

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()
