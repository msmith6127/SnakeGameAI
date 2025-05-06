from agent import Agent
from SnakeGame import SnakeGame
import matplotlib.pyplot as plt

# Optional: for plotting scores
def plot(scores, mean_scores, survival_times=None, mean_survival_times=None, losses=None, exploration_rates=None):
    '''
    This is for visualizing progress; 
    plotting the game score, average scores, survival times, losses and exploration rates during training
    '''
    
    plt.clf()
    
    # check the number of subplots needed
    rows = 2  # Minimum number of plots
    if survival_times is not None:
        rows += 1
    if losses is not None:
        rows += 1
    if exploration_rates is not None:
        rows += 1
    
    current_row = 1
    
    # plot scores
    plt.subplot(rows, 1, current_row)
    plt.title('Training...')
    plt.xlabel('Game')
    plt.ylabel('Score')
    plt.plot(scores, label='Score')
    plt.plot(mean_scores, label='Mean Score')
    plt.ylim(ymin=0)
    if len(scores) > 0:
        plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    if len(mean_scores) > 0:
        plt.text(len(mean_scores) - 1, mean_scores[-1], str(round(mean_scores[-1], 2)))
    plt.legend()
    
    current_row += 1
    
    # plot survival times
    if survival_times is not None and mean_survival_times is not None:
        plt.subplot(rows, 1, current_row)
        plt.title('Survival Time')
        plt.xlabel('Game')
        plt.ylabel('Frames')
        plt.plot(survival_times, label='Survival Time', color='green')
        plt.plot(mean_survival_times, label='Mean Survival Time', color='olive')
        plt.ylim(ymin=0)
        if len(survival_times) > 0:
            plt.text(len(survival_times) - 1, survival_times[-1], str(survival_times[-1]))
        if len(mean_survival_times) > 0:
            plt.text(len(mean_survival_times) - 1, mean_survival_times[-1], str(round(mean_survival_times[-1], 2)))
        plt.legend()
        current_row += 1
    
    # plot losses 
    if losses is not None:
        plt.subplot(rows, 1, current_row)
        plt.title('Training Loss')
        plt.xlabel('Game')
        plt.ylabel('Loss')
        plt.plot(losses, label='Loss', color='red')
        plt.ylim(ymin=0)
        if len(losses) > 0:
            plt.text(len(losses) - 1, losses[-1], str(round(losses[-1], 4)))
        plt.legend()
        current_row += 1
    
    # plot exploration rates 
    if exploration_rates is not None:
        plt.subplot(rows, 1, current_row)
        plt.title('Exploration-Exploitation Balance')
        plt.xlabel('Game')
        plt.ylabel('Exploration Rate')
        plt.plot(exploration_rates, label='Exploration Rate', color='purple')
        plt.ylim(0, 0.5)  # Epsilon/200 ranges from 0 to 0.4
        if len(exploration_rates) > 0:
            plt.text(len(exploration_rates) - 1, exploration_rates[-1], str(round(exploration_rates[-1], 4)))
        plt.legend()
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5) 
    plt.pause(0.1)

def train():
    '''
    - Plays the game on loop (repeatedly).
    - Agent decides moves based on experience which will help improve over time. 
    - Scores progress as training continues.
    '''
    plot_scores = []
    plot_mean_scores = []
    plot_survival_times = []
    plot_mean_survival_times = []
    plot_losses = [] 
    plot_exploration_rates = []
    total_score = 0
    total_survival_time = 0
    record = 0
    
    plt.figure(figsize=(10, 10))

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
            
            #track survival time
            survival_time = game.frame_iteration
            agent.survival_times.append(survival_time)

            # Game over - reset and train long memory
            game.reset()
            agent.n_games += 1
            long_memory_loss = agent.train_long_memory()
            
            # calculate exploration rate
            current_epsilon = max(0, 80 - agent.n_games)
            exploration_rate = current_epsilon / 200
            agent.exploration_rates.append(exploration_rate)

            # Track high score
            if score > record:
                record = score
                agent.model.save()
                
            # calculate exploration percentage for display
            total_actions = agent.random_actions + agent.model_actions
            exploration_percentage = (agent.random_actions / total_actions * 100) if total_actions > 0 else 0
            exploitation_percentage = 100 - exploration_percentage

            print(f'Game {agent.n_games} - Score: {score} - Record: {record} - Survival Time: {survival_time} '
                  f'- Loss: {round(long_memory_loss, 4)} - Explore/Exploit: {round(exploration_percentage, 1)}%/{round(exploitation_percentage, 1)}%')

            # update plots
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            
            # track survival times
            plot_survival_times.append(survival_time)
            total_survival_time += survival_time
            mean_survival_time = total_survival_time / agent.n_games
            plot_mean_survival_times.append(mean_survival_time)
            
            # track losses
            plot_losses.append(long_memory_loss)
            
            # track exploration rates
            plot_exploration_rates.append(exploration_rate)
            
            # plot all
            plot(plot_scores, plot_mean_scores, plot_survival_times, plot_mean_survival_times, plot_losses, plot_exploration_rates)

if __name__ == '__main__':
    train()
