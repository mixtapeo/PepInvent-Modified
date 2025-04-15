import os
import json
from typing import List, Dict, Tuple, Set
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import pandas as pd
from collections import defaultdict

class PeptideValidator:
    def __init__(self, natural_aa_set=None, training_nnaa_set=None):
        """
        Initialize the validator with reference sets.
        
        Args:
            natural_aa_set: Set of SMILES strings for natural amino acids
            training_nnaa_set: Set of SMILES strings for non-natural amino acids used in training
        """
        # Default set of natural amino acids if not provided
        self.natural_aa_set = natural_aa_set or self._get_default_natural_aa()
        self.training_nnaa_set = training_nnaa_set or set()
    
    def _get_default_natural_aa(self) -> Set[str]:
        """Returns a set of SMILES for the 20 standard amino acids in CHUCKLES format."""
        # Simplified representation - would need exact CHUCKLES format in practice
        return {
            "N[C@@H](C)C(=O)",  # Alanine (A)
            "N[C@@H](CC(=O)N)C(=O)",  # Asparagine (N)
            # Other amino acids would be listed here
        }
    
    def validate_peptides(self, generated_peptides: List[str]) -> Dict:
        """
        Comprehensive validation following PepINVENT paper methods.
        """
        results = {
            'peptides': [],
            'validity': {'valid_count': 0, 'invalid_count': 0, 'parse_error_count': 0},
            'uniqueness': {
                'peptide_level': set(),
                'amino_acid_level': {
                    'string_level': set(),
                    'isomeric_smiles_level': set(),
                    'canonical_smiles_level': set()
                }
            },
            'novelty': {
                'natural': set(),
                'non_natural': set(),
                'novel': set()
            },
            'topology': defaultdict(int)
        }
        
        for peptide in generated_peptides:
            # Basic validity check
            try:
                mol = Chem.MolFromSmiles(peptide)
                if mol is None:
                    results['validity']['invalid_count'] += 1
                    results['peptides'].append({'smiles': peptide, 'valid': False})
                    continue
            except Exception as e:
                results['validity']['parse_error_count'] += 1
                continue
            
            # Canonicalize for uniqueness checks
            canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            results['uniqueness']['peptide_level'].add(canonical_smiles)
            
            # Extract amino acids
            amino_acids = peptide.split('|') if '|' in peptide else []
            
            # Topology validation
            try:
                # Get ring information
                rings = mol.GetRingInfo().AtomRings()  # Using GetRingInfo() instead of GetSSSR()
                largest_ring = 0
                for ring in rings:
                    if len(ring) > largest_ring:
                        largest_ring = len(ring)
                    
                if largest_ring == 0:
                    topology = "linear"
                elif largest_ring >= 12:  # Macrocycle definition from paper
                    topology = "macrocyclic"
                else:
                    topology = "other_cyclic"
                
                results['topology'][topology] += 1
            except Exception as e:
                # If ring detection fails, count it as invalid
                results['validity']['invalid_count'] += 1
                continue
            
            # Check peptide bonds
            try:
                amide_bonds = sum(1 for bond in mol.GetBonds() 
                               if self._is_amide_bond(bond))
                
                if amide_bonds < len(amino_acids)-2 and len(amino_acids) > 1:  # Expected peptide bonds
                    results['validity']['invalid_count'] += 1
                    continue
            except Exception as e:
                results['validity']['invalid_count'] += 1
                continue
                
            # Passed all checks - peptide is valid
            results['validity']['valid_count'] += 1
            results['peptides'].append({'smiles': peptide, 'valid': True})
            
            # Process amino acids for uniqueness and novelty
            for aa in amino_acids:
                if not aa or aa == "?":  # Skip empty or masked positions
                    continue
                    
                # Add to string-level uniqueness
                results['uniqueness']['amino_acid_level']['string_level'].add(aa)
                
                # Process with RDKit
                try:
                    aa_mol = Chem.MolFromSmiles(aa)
                    if aa_mol:
                        # Isomeric SMILES (with stereochemistry)
                        iso = Chem.MolToSmiles(aa_mol, isomericSmiles=True)
                        results['uniqueness']['amino_acid_level']['isomeric_smiles_level'].add(iso)
                        
                        # Canonical SMILES (without stereochemistry)
                        canon = Chem.MolToSmiles(aa_mol, isomericSmiles=False)
                        results['uniqueness']['amino_acid_level']['canonical_smiles_level'].add(canon)
                        
                        # Novelty check
                        if aa in self.natural_aa_set:
                            results['novelty']['natural'].add(aa)
                        elif aa in self.training_nnaa_set:
                            results['novelty']['non_natural'].add(aa)
                        else:
                            results['novelty']['novel'].add(aa)
                except Exception as e:
                    continue
        
        # Convert uniqueness counts
        for key in results['uniqueness']['amino_acid_level']:
            results['uniqueness']['amino_acid_level'][key] = len(
                results['uniqueness']['amino_acid_level'][key]
            )
        
        results['uniqueness']['peptide_level'] = len(results['uniqueness']['peptide_level'])
        
        #  Convert novelty counts
        for key in list(results['novelty'].keys()):  # Use list() to create a static copy of keys
            results['novelty'][f'{key}_count'] = len(results['novelty'][key])
            results['novelty'][key] = list(results['novelty'][key])
        
        return results
    
    def _is_amide_bond(self, bond):
        """Detects peptide bonds (amide linkages) between amino acids."""
        try:
            if bond.GetBondType() != Chem.BondType.SINGLE:
                return False
                
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()
            
            # Check C-N bond where C is part of carbonyl
            if (begin_atom.GetSymbol() == 'C' and end_atom.GetSymbol() == 'N'):
                # Check if carbon is part of carbonyl
                for neighbor in begin_atom.GetNeighbors():
                    if neighbor.GetSymbol() == 'O' and neighbor != end_atom:
                        for b in neighbor.GetBonds():
                            if b.GetBondType() == Chem.BondType.DOUBLE:
                                return True
                                
            # Check N-C direction
            elif (begin_atom.GetSymbol() == 'N' and end_atom.GetSymbol() == 'C'):
                # Check if carbon is part of carbonyl
                for neighbor in end_atom.GetNeighbors():
                    if neighbor.GetSymbol() == 'O' and neighbor != begin_atom:
                        for b in neighbor.GetBonds():
                            if b.GetBondType() == Chem.BondType.DOUBLE:
                                return True
        except Exception:
            pass
                                
        return False

def run_validation(output_path):
    """Run validation on results from a sampling run."""
    print(f"Loading data from {output_path}")
    
    try:
        results_df = pd.read_csv(output_path)
        
        # Extract generated peptides from results
        generated_peptides = []
        for col in [c for c in results_df.columns if c.startswith('Generated_smi_')]:
            peptides = results_df[col].tolist()
            generated_peptides.extend([p for p in peptides if isinstance(p, str) and p.strip()])
        
        print(f"Found {len(generated_peptides)} peptides across all columns")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None
    
    # Filter out None values and empty strings
    generated_peptides = [p for p in generated_peptides if isinstance(p, str) and p.strip()]
    
    print(f"Processing {len(generated_peptides)} valid peptide strings")
    
    # Run validation
    validator = PeptideValidator()
    results = validator.validate_peptides(generated_peptides)
    
    # Print summary
    print("\n--- VALIDATION RESULTS ---")
    print(f"Total peptides: {len(generated_peptides)}")
    total_valid = results['validity']['valid_count']
    total_invalid = results['validity']['invalid_count']
    parse_errors = results['validity']['parse_error_count']
    
    print(f"Valid peptides: {total_valid} ({total_valid/len(generated_peptides)*100:.1f}%)")
    print(f"Invalid peptides: {total_invalid} ({total_invalid/len(generated_peptides)*100:.1f}%)")
    print(f"Parse errors: {parse_errors} ({parse_errors/len(generated_peptides)*100:.1f}%)")
    print(f"Unique peptides: {results['uniqueness']['peptide_level']}")
    
    print("\nAmino acid uniqueness:")
    for level, count in results['uniqueness']['amino_acid_level'].items():
        print(f"  {level}: {count}")
    
    print("\nNovelty:")
    print(f"  Natural amino acids: {results['novelty']['natural_count']}")
    print(f"  Non-natural amino acids: {results['novelty']['non_natural_count']}")
    print(f"  Novel amino acids: {results['novelty']['novel_count']}")
    
    print("\nTopology distribution:")
    for topology, count in results['topology'].items():
        print(f"  {topology}: {count}")
    
    # Save detailed results to a file
    try:
        with open(output_path.replace('.csv', '_validation.json'), 'w') as f:
            # Convert sets to lists for JSON serialization
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to {output_path.replace('.csv', '_validation.json')}")
    except Exception as e:
        print(f"Error saving results: {e}")
    
    return results

if __name__ == "__main__":
    # Path to the sampling results
    results_path = "./results/sampling_output.csv"
    
    # Run validation if file exists
    if os.path.exists(results_path):
        run_validation(results_path)
    else:
        print(f"Results file not found: {results_path}")