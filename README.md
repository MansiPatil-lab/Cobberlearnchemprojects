import pubchempy as pcp

# Ask the user for a compound name
compound_name = input("Enter a compound name: ")

# Search PubChem by name
results = pcp.get_compounds(compound_name, "name")

# Check if the compound was found
if results:
    compound = results[0]

    print("\n" + "=" * 45)
    print(f"        COMPOUND INFORMATION")
    print("=" * 45)

    print(f"Name:              {compound_name}")
    print(f"PubChem CID:       {compound.cid}")
    print(f"Molecular Formula: {compound.molecular_formula}")
    print(f"Molecular Weight:  {compound.molecular_weight}")
    print(f"SMILES:            {compound.smiles}")
    print(f"IUPAC Name:        {compound.iupac_name}")
    print(f"XLogP:             {compound.xlogp}")
    print(f"TPSA:              {compound.tpsa}")
    print(f"H-Bond Donors:     {compound.h_bond_donor_count}")
    print(f"H-Bond Acceptors:  {compound.h_bond_acceptor_count}")
    print(f"Rotatable Bonds:   {compound.rotatable_bond_count}")
    print(f"Heavy Atoms:       {compound.heavy_atom_count}")

    print("=" * 45)

else:
    print(f"\nSorry, '{compound_name}' was not found.")