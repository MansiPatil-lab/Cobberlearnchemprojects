from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

print("=" * 50)
print("       MOLECULAR PROPERTY CALCULATOR")
print("=" * 50)

while True:

    # Ask the user for a molecule
    name = input("\nEnter the molecule name (or type 'quit' to exit): ")

    if name.lower() == "quit":
        print("\nGoodbye!")
        break

    # Ask for the SMILES string
    smiles = input("Enter the SMILES string: ")

    # Create the molecule object
    molecule = Chem.MolFromSmiles(smiles)

    # Check if the SMILES string is valid
    if molecule is None:
        print("\nInvalid SMILES string. Please try again.")
        continue

    # Calculate molecular descriptors
    exact_weight = Descriptors.ExactMolWt(molecule)
    h_bond_donors = Lipinski.NumHDonors(molecule)
    tpsa = Descriptors.TPSA(molecule)

    # Two additional descriptors
    logp = Descriptors.MolLogP(molecule)
    rotatable_bonds = Descriptors.NumRotatableBonds(molecule)

    # Display the results
    print("\n" + "=" * 50)
    print(f"       RESULTS FOR {name.upper()}")
    print("=" * 50)
    print(f"SMILES:                    {smiles}")
    print(f"Exact Molecular Weight:    {exact_weight:.4f}")
    print(f"Hydrogen Bond Donors:      {h_bond_donors}")
    print(f"TPSA:                      {tpsa:.2f} Å²")
    print(f"LogP:                      {logp:.2f}")
    print(f"Rotatable Bonds:            {rotatable_bonds}")
    print("=" * 50)