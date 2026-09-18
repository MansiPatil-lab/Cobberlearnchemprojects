import pubchempy as pcp


def main():
    results = pcp.get_compounds("theobromine", "name")
    if not results:
        raise RuntimeError("No PubChem compound was found for theobromine.")

    compound = results[0]
    print("Theobromine")
    print("--------------------")
    print("Molecular Weight:", compound.molecular_weight)
    print("Molecular Formula:", compound.molecular_formula)
    print("SMILES:", compound.smiles)


if __name__ == "__main__":
    main()
