# Anirudh Sridhar, Carleton University.
# This file validates outputted SMILES strings.
# Take input smile string, replace ? with output's smile string, and validate. Each output's result is seperated by |.

# notes:
# example: one valid peptide from default RL run:
# C1=C(NC=N1)C[C@@H](C(=O)O)NC(=O)CN[C@@H](C[C@@H]1C[C@@]21CCCC12CCCC1)C(=O)C(=O)[C@H](CCCCN[C@@H]([C@@H](C)O)C(=O)O)
# target mol: desired partner to generated ?s

from rdkit import Chem
from pepinvent.scoring_function.scoring_components.predictive_model import PredictiveModel
from pepinvent.scoring_function.scoring_components.scoring_component_parameters import ScoringComponentParameters
import os
import pickle
from rdkit.Chem import AllChem
import numpy as np

def predict_permeability(peptide_smiles):
    """
    PepINVENT researches created a privately sourced permeability model.
    This loads the model and calls its prediction function.
    
    Args:
        peptide_smiles (str): SMILES string of the peptide
        
    Returns:
        float: Probability of the peptide being permeable (0-1)
    """
    try:
        # Load model and scalar directly with pickle
        model_path = os.path.join(os.getcwd(), "models", "predictive_model.pckl")
        scalar_path = os.path.join(os.getcwd(), "models", "feature_scalar.pckl")
        
        model = pickle.load(open(model_path, 'rb'))
        scalar = pickle.load(open(scalar_path, 'rb'))
        
        # Generate fingerprint for the molecule
        mol = Chem.MolFromSmiles(peptide_smiles)
        if mol is None:
            return None
            
        # Calculate Morgan fingerprint
        fps = AllChem.GetMorganFingerprint(mol, radius=4, useChirality=True, useCounts=True)
        
        # Convert to array format
        size = 2048  # Common fingerprint size
        arr = np.zeros((size,), np.int32)
        for idx, v in fps.GetNonzeroElements().items():
            nidx = idx % size
            arr[nidx] += int(v)
        
        # Transform using scalar
        x_test = scalar.transform([arr])
        
        # Make prediction
        predictions = model.predict_proba(x_test)
        return predictions[0, 1]  # Return probability of positive class
        
    except Exception as e:
        print(f"Error predicting permeability: {str(e)}")
        return None

def validate_peptide_smiles(template_smiles, generated_peptides, detailed=False):
    """
    validate if combining the template SMILES with generated peptide SMILES creates valid molecules.
    
    args:
        template_smiles (str): Template SMILES string with '?' placeholders.
        generated_peptides (str): Generated peptide fragments separated by '|'.
        detailed (bool): Whether to return detailed error information.
        
    Returns:
        list: List of dictionaries with validation results.
    """
    results = []
    
    # Split template by the '|' delimiter to get parts (ensure the template is not pre-split)
    # In this approach our template is a single string containing '?' where fragments need to be inserted.
    # We replace each occurrence of '?' in sequence with the provided generated fragments.
    peptide_parts = generated_peptides.split('|')
    temp_template = template_smiles
    for part in peptide_parts:
        temp_template = temp_template.replace('?', part.strip(), 1)
    
    # Validate the combined SMILES
    mol = Chem.MolFromSmiles(temp_template)
    is_valid = mol is not None
    error_msg = None
    
    if not is_valid and detailed:
        try:
            problematic_mol = Chem.MolFromSmiles(temp_template, sanitize=False)
            if problematic_mol:
                try:
                    Chem.SanitizeMol(problematic_mol)
                except Exception as e:
                    error_msg = str(e)
            else:
                error_msg = "Invalid SMILES syntax"
        except Exception as e:
            error_msg = "Invalid SMILES syntax"
        
    result = {
        "template_smiles": template_smiles,
        "generated_parts": peptide_parts,
        "combined_smiles": temp_template,
        "is_valid": is_valid
    }

    if detailed and not is_valid:
        result["error"] = error_msg

    results.append(result)
    return results

def validate_peptide_generation(source_mol, generated_peptide, detailed=False):
    """
    Validate a peptide generation by replacing '?' placeholders in source_mol with fragments from generated_peptide.
    
    Args:
        source_mol (str): Source molecule template with '?' placeholders.
        generated_peptide (str): Generated peptide fragments separated by '|'.
        detailed (bool): Whether to provide detailed error information.
        
    Returns:
        dict: Validation results along with summary of replacement.
    """
    results = validate_peptide_smiles(source_mol, generated_peptide, detailed)
    
    placeholder_count = source_mol.count('?')
    generated_parts = generated_peptide.split('|')
    
    summary = {
        "source_template": source_mol,
        "generated_peptide": generated_peptide,
        "placeholder_count": placeholder_count,
        "generated_parts_count": len(generated_parts),
        "validation_results": results
    }
    
    return summary

if __name__ == "__main__":
    # Define a source molecule template (with '?' placeholders)
    source_mol = "N[C@@H](?)C(=O)O"    
    # Define generated peptide fragments to replace the '?'
    generated_peptide = "C1=CC=CC=C1"  # e.g. benzene ring fragment
    
    # Validate the generation
    validation = validate_peptide_generation(source_mol, generated_peptide, detailed=True)

    print("Validation Results:")
    for result in validation["validation_results"]:
        if result["is_valid"]:
            print("VALID: The combined peptide is valid")
            print(f"Combined SMILES: {result['combined_smiles']}")
            print("---------------------------------------")
            # perm = predict_permeability(result['combined_smiles'])
            # print(f"Permeability Probability: {perm}")
        else:
            print("INVALID: The combined peptide is not valid")
            print(result['combined_smiles'])
    
    # test
    # median / low permiability: -6.2 from cycpeptmpdb
    print(predict_permeability(" 	CC(C)C[C@@H]1NC(=O)[C@@H](CC(C)C)NC(=O)[C@@H](CC(C)C)NC(=O)[C@H](Cc2ccc(O)cc2)NC(=O)[C@@H]2CCCN2C(=O)[C@@H](CC(C)C)NC1=O"))

    # high -4.00
    print(predict_permeability("CC(C)C[C@H]1C(=O)N[C@@H](Cc2ccccc2)C(=O)N(C)[C@@H](C)C(=O)N[C@H](CC(C)C)C(=O)N(C)[C@H](CC(C)C)C(=O)N2CCC[C@H]2C(=O)N1C"))

    #very low -10
    print(predict_permeability("CC(C)C[C@@H]1NC(=O)CNC(=O)[C@@H]2CCCN2C(=O)[C@H](Cc2ccccc2)NC(=O)CNC1=O"))