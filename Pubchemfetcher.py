import pubchempy as pcp

# Fetch theobromine by name
results = pcp.get_compounds("theobromine", "name")

# Get the first result
compound = results[0]

# Print the requested information
print("Theobromine")
print("--------------------")
print("Molecular Weight:", compound.molecular_weight)
print("Molecular Formula:", compound.molecular_formula)
print("SMILES:", compound.smiles)