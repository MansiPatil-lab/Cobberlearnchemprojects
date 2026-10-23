Chapter 3 — Your Digital Lab: Learning to Work Like a Modern Scientist
This chapter was about getting familiar with the tools we'll use throughout the course, like Python, PyCharm, GitHub, and Overleaf. Before this, I didn't really think of GitHub as a lab notebook, but now I see why keeping my code organized matters. I also learned that AI can help me get started when I'm stuck, but I still have to understand and check whatever code I use.

Chapter 4 — Your Work, Your Words: Portfolios That Speak for Your Learning
Chapter 4 made me think about how I present my work. Having a Python file is one thing, but someone looking at my GitHub should also be able to understand what I was trying to do. That's why I'm writing this README. I also like that the course separates the coding portfolio from the ethics portfolio, because learning how to use AI and thinking about its effects are two different parts of the same experience.

Chapter 5 — From Click to Code: Accessing Chemical Data with Confidence
Here I learned more about where chemical data comes from. We looked at databases like PubChem and the Protein Data Bank, along with ways to retrieve information using code. I started to understand why details like a chemical identifier or a SMILES string are important. If I pull information for the wrong molecule, it doesn't matter how good my later analysis looks.

Chapter 6 — Patterns Among Molecules: Exploring Chemical Similarity with ML Tools
I used to think comparing molecules mainly meant looking at their structures. This chapter showed me that computers also compare them using numbers and chemical properties. We covered things like molecular weight, polar surface area, and using RDKit to work with molecular information. What stood out to me is that choosing which properties to compare can really affect what we decide is “similar.”

Chapter 7 — Mind the Gaps: Learning to Impute with Insight and Integrity
Missing data seemed like a small problem at first, but this chapter helped me see why it matters. Just deleting incomplete rows can mean throwing away useful information. We learned about imputation, which is a way to estimate missing values, and why those estimates need to be handled carefully. I want to remember this when I work with real assay data, especially because a missing result isn't the same thing as an inactive chemical.

Chapter 8 — Judging the Fit: A Practical Guide to Error Analysis
This was about checking how well a model is actually doing. A graph can look convincing, but that doesn't necessarily mean the predictions are accurate. We went over residuals and measurements like MAE, MSE, and R². I'm learning that getting an answer from a model is only part of the job; I also need to ask how far off it might be and whether it was tested fairly.

Chapter 9 — Branch by Branch: Teaching Trees to Make Smarter Predictions
Decision trees were easier for me to picture because they work through questions and branches. Each question helps the model make a decision based on the data. The chapter connects this to predicting chemical properties, including water solubility. It also introduced an important issue: a model can become so complicated that it fits its training examples well but doesn't do as well on new ones.

Chapter 10 — Clustering and Classification: Mapping Meaning in Chemical Data
This chapter helped me sort out two ideas I kept mixing up: clustering and classification. Clustering looks for groups in data without being told the answers first, while classification learns from examples that already have labels. We explored K-means and K-nearest neighbors, including connections to periodic trends. It made me realize that finding a pattern isn't enough—I still need to think about whether the pattern makes sense chemically.

Chapter 11 — Step by Step: Learning Through Gradient Descent
Gradient descent finally made more sense when I thought of it as making small changes and checking whether the error gets better or worse. The goal is to lower the loss instead of trying to guess the perfect settings all at once. The chapter also showed why the size of each step matters. Too big of a step can miss a good answer, and very small steps can take a long time.

Chapter 12 — Breeding Better Corn: Evolutionary Algorithms at Work
I thought this was an interesting way to approach problems because it works differently from gradient descent. Evolutionary algorithms try out a group of possible solutions, keep some of the better ones, and create new possibilities over several generations. We connected that idea to searching for useful molecules. I liked seeing that machine learning and optimization don't always have to follow one single method.

Chapter 13 — Learning in Layers: Understanding Artificial Neural Networks
This chapter took some extra thinking for me. We learned about neurons, weights, biases, activation functions, and hidden layers. The sigmoid examples helped me understand why one simple curve can't fit every complicated pattern and why adding neurons can help. At the same time, more neurons don't automatically mean a better model. I also learned to pay attention to the loss during training and to the risk of overfitting.
