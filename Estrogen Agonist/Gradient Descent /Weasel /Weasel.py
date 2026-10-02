import random
import string
import os
import matplotlib.pyplot as plt

# Target phrase
TARGET = "METHINKS IT IS LIKE A WEASEL"

# -----------------------------
# Fitness function
# -----------------------------
def fitness(candidate):
    score = 0

    for i in range(len(TARGET)):
        if candidate[i] == TARGET[i]:
            score += 1

    return score


# -----------------------------
# Mutation function
# -----------------------------
def mutate_string(text):
    characters = string.ascii_uppercase + " "
    mutated = ""

    for character in text:
        if random.random() < 0.05:   # 5% mutation chance
            mutated += random.choice(characters)
        else:
            mutated += character

    return mutated


# -----------------------------
# Generate random starting string
# -----------------------------
characters = string.ascii_uppercase + " "

parent = ""

for i in range(len(TARGET)):
    parent += random.choice(characters)

best_score = fitness(parent)

# Keep track of fitness over time
generations = [0]
fitness_scores = [best_score]

# Keep track of every generation
generation_history = []

generation_history.append(
    f"Generation 0 | Score: {best_score} | {parent}\n"
)

print("Generation:", 0)
print("Best score:", best_score)
print("Best string:", parent)
print()


# -----------------------------
# Natural selection
# -----------------------------
generation = 0

while best_score < len(TARGET):

    generation += 1

    best_offspring = None
    best_offspring_score = -1

    # Create 100 offspring
    for i in range(100):

        offspring = mutate_string(parent)
        offspring_score = fitness(offspring)

        # Keep the best offspring
        if offspring_score > best_offspring_score:
            best_offspring = offspring
            best_offspring_score = offspring_score

    # Replace parent only if offspring is better
    if best_offspring_score > best_score:
        parent = best_offspring
        best_score = best_offspring_score

    # Save fitness information
    generations.append(generation)
    fitness_scores.append(best_score)

    # Save generation information
    generation_history.append(
        f"Generation {generation} | Score: {best_score} | {parent}\n"
    )

    # Print progress
    print("Generation:", generation)
    print("Best score:", best_score)
    print("Best string:", parent)
    print()


# -----------------------------
# Save all generations to file
# -----------------------------

script_directory = os.path.dirname(os.path.abspath(__file__))

history_file = os.path.join(
    script_directory,
    "weasel_generations.txt"
)

with open(history_file, "w") as file:
    file.writelines(generation_history)

print("Generation history saved to:")
print(history_file)


# -----------------------------
# Plot fitness over time
# -----------------------------

plt.figure(figsize=(8, 5))

plt.plot(generations, fitness_scores)

plt.xlabel("Generation")
plt.ylabel("Fitness Score")
plt.title("Weasel Program: Fitness Over Time")

plt.ylim(0, 28)
plt.grid(True)

# Save the graph automatically
plot_file = os.path.join(
    script_directory,
    "weasel_fitness.png"
)

plt.savefig(plot_file, dpi=300, bbox_inches="tight")

print("Fitness plot saved to:")
print(plot_file)

plt.show()