import random
import string

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
        if random.random() < 0.05:   # 5% chance of mutation
            mutated += random.choice(characters)
        else:
            mutated += character

    return mutated


# -----------------------------
# Generate a random starting string
# -----------------------------
characters = string.ascii_uppercase + " "

parent = ""

for i in range(len(TARGET)):
    parent += random.choice(characters)

# Score the starting string
best_score = fitness(parent)

print("Generation:", 0)
print("Best score:", best_score)
print("Best string:", parent)
print()


# -----------------------------
# Natural selection loop
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

    # Print progress
    print("Generation:", generation)
    print("Best score:", best_score)
    print("Best string:", parent)
    print()