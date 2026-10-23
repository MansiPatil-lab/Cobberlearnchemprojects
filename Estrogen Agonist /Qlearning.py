
import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess

# STEP 1: Create FrozenLake environment
env = gym.make("FrozenLake-v1")

num_states = env.observation_space.n
num_actions = env.action_space.n

# STEP 2: Create the Q-table
q_table = np.zeros((num_states, num_actions))

print("Initial Q-Table:")
print(q_table)
print("Q-Table Shape:", q_table.shape)

# STEP 3: Training settings
episodes = 50000
learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0
min_epsilon = 0.05
epsilon_decay = 0.9999

# STEP 4: Train the agent
for episode in range(episodes):

    state, info = env.reset()
    done = False

    while not done:

        # Explore or use learned knowledge
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(q_table[state]))

        # Take action
        new_state, reward, terminated, truncated, info = (
            env.step(action)
        )

        done = terminated or truncated

        # Bellman equation
        if done:
            target = reward
        else:
            target = reward + discount_factor * np.max(
                q_table[new_state]
            )

        # Update Q-table
        q_table[state, action] += learning_rate * (
            target - q_table[state, action]
        )

        state = new_state

    epsilon = max(min_epsilon, epsilon * epsilon_decay)

print("\nTraining Completed!")

# STEP 5: Print final Q-table
np.set_printoptions(precision=4, suppress=True)
print("\nFinal Q-Table:")
print(q_table)

# STEP 6: Create learned policy with arrows
arrows = ["←", "↓", "→", "↑"]

lake = env.unwrapped.desc
rows, cols = lake.shape

def create_policy(q_table):
    policy = []

    for row in range(rows):
        policy_row = []

        for col in range(cols):
            state = row * cols + col
            cell = lake[row, col].decode("utf-8")

            if cell == "H":
                symbol = "H"
            elif cell == "G":
                symbol = "G"
            elif np.allclose(q_table[state], 0):
                symbol = "?"
            else:
                best_action = np.argmax(q_table[state])
                symbol = arrows[best_action]

            policy_row.append(symbol)

        policy.append(policy_row)

    return policy

policy = create_policy(q_table)

print("\nLearned Policy Grid:")

for row in policy:
    print("  ".join(row))

# STEP 7: Test agent for 100 episodes
test_episodes = 100
successes = 0

for episode in range(test_episodes):

    state, info = env.reset()
    done = False

    while not done:

        # No exploration during testing
        action = int(np.argmax(q_table[state]))

        new_state, reward, terminated, truncated, info = (
            env.step(action)
        )

        state = new_state
        done = terminated or truncated

        if reward == 1:
            successes += 1

# STEP 8: Calculate success rate
success_rate = (successes / test_episodes) * 100

print("\nTEST RESULTS")
print("Test Episodes:", test_episodes)
print("Successful Episodes:", successes)
print(f"Success Rate: {success_rate:.2f}%")

if success_rate > 70:
    print("Good learning!")
elif success_rate < 30:
    print("The agent needs more training.")
else:
    print("The agent shows some learning.")

# STEP 9: Create a policy image
fig, ax = plt.subplots(figsize=(6, 6))

colors = {
    "S": "lightblue",
    "F": "white",
    "H": "lightgray",
    "G": "lightgreen"
}

for row in range(rows):
    for col in range(cols):

        cell = lake[row, col].decode("utf-8")

        rect = plt.Rectangle(
            (col, row),
            1, 1,
            facecolor=colors[cell],
            edgecolor="black"
        )

        ax.add_patch(rect)

        symbol = policy[row][col]

        if cell == "S":
            symbol = "S\n" + symbol

        ax.text(
            col + 0.5,
            row + 0.5,
            symbol,
            ha="center",
            va="center",
            fontsize=20
        )

ax.set_xlim(0, cols)
ax.set_ylim(rows, 0)
ax.set_aspect("equal")
ax.axis("off")

ax.set_title(
    f"FrozenLake Learned Policy\n"
    f"Success Rate: {success_rate:.1f}%"
)

plt.tight_layout()

# STEP 10: Save image
output_file = (
    Path(__file__).resolve().parent
    / "FrozenLake_Learned_Policy.png"
)

fig.savefig(
    str(output_file),
    format="png",
    dpi=300
)

plt.close(fig)
env.close()

print("\nImage saved successfully!")
print("Saved to:", output_file)

# Automatically open image on Mac
subprocess.run(["open", str(output_file)])
