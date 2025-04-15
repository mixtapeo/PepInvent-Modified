import os
import json
import argparse
from typing import Dict, List
import pandas as pd
import tempfile
from rdkit import Chem
from rdkit.Chem import AllChem

from validator import PeptideValidator
from pep_structure_analyzer import PepStructureAnalyzer, extract_peptide_binding_pairs

class PepINVENTStructureWorkflow:
    """Workflow integrating PepINVENT generation with structural analysis."""
    
    def __init__(self):
        """Initialize workflow components."""
        self.validator = PeptideValidator()
        self.analyzer = PepStructureAnalyzer()
        
    def modify_peptide(self, pdb_id: str, modifications: Dict, output_dir: str) -> Dict:
        """
        Modify a peptide from PDB and analyze the modifications.
        
        Args:
            pdb_id: PDB ID containing the peptide
            modifications: Dict specifying modifications to make
            output_dir: Directory to save output files
        
        Returns:
            Dict with results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract peptide-protein binding pair
        binding_pair = extract_peptide_binding_pairs(pdb_id, output_dir)
        if 'error' in binding_pair:
            return binding_pair
            
        peptide_seq = binding_pair["peptide_sequence"]
        print(f"Original peptide: {peptide_seq}")
        
        # Create modified peptide sequence
        modified_seq = self._apply_modifications(peptide_seq, modifications)
        print(f"Modified peptide: {modified_seq}")
        
        # Save modified peptide details
        peptide_data = {
            "pdb_id": pdb_id,
            "original_sequence": peptide_seq,
            "modified_sequence": modified_seq,
            "modifications": modifications
        }
        
        with open(os.path.join(output_dir, "modification_details.json"), "w") as f:
            json.dump(peptide_data, f, indent=2)
            
        # Generate 3D structure for modified peptide
        modified_pdb_path = self._generate_3d_structure(modified_seq, output_dir)
        
        # Compare structures
        structure_comparison = self.analyzer.compare_structures(
            binding_pair["peptide_path"],
            modified_pdb_path
        )
        
        # Estimate binding affinity for original
        original_affinity = self.analyzer.estimate_binding_affinity(
            binding_pair["receptor_path"], 
            binding_pair["peptide_path"]
        )
        
        # Estimate binding affinity for modified peptide
        modified_affinity = self.analyzer.estimate_binding_affinity(
            binding_pair["receptor_path"], 
            modified_pdb_path
        )
        
        # Cell penetration predictions
        original_cpp = self.analyzer.predict_cell_penetration(peptide_seq)
        modified_cpp = self.analyzer.predict_cell_penetration(modified_seq)
        
        # Compile results
        results = {
            "pdb_id": pdb_id,
            "original_peptide": {
                "sequence": peptide_seq,
                "path": binding_pair["peptide_path"],
                "binding_affinity": original_affinity,
                "cell_penetration": original_cpp
            },
            "modified_peptide": {
                "sequence": modified_seq,
                "path": modified_pdb_path,
                "binding_affinity": modified_affinity,
                "cell_penetration": modified_cpp
            },
            "structure_comparison": structure_comparison
        }
        
        # Save final results
        with open(os.path.join(output_dir, "analysis_results.json"), "w") as f:
            json.dump(results, f, indent=2)
            
        return results
    
    def _apply_modifications(self, peptide_seq: str, modifications: Dict) -> str:
        """Apply modifications to peptide sequence."""
        modified_seq = list(peptide_seq)
        
        # Position-specific substitutions
        for position, new_aa in modifications.get("substitutions", {}).items():
            pos = int(position)
            if 0 <= pos < len(modified_seq):
                modified_seq[pos] = new_aa
        
        # Insertions (from N to C terminus)
        for position, aa_to_insert in modifications.get("insertions", {}).items():
            pos = int(position)
            if 0 <= pos <= len(modified_seq):
                modified_seq.insert(pos, aa_to_insert)
                
        # Deletions (from N to C terminus)
        for position in sorted(modifications.get("deletions", []), reverse=True):
            pos = int(position)
            if 0 <= pos < len(modified_seq):
                modified_seq.pop(pos)
        
        return "".join(modified_seq)
    
    def _generate_3d_structure(self, peptide_seq: str, output_dir: str) -> str:
        """Generate 3D structure for peptide sequence."""
        # This is a simplified approach - in practice you'd want to use more 
        # sophisticated methods like Rosetta or AlphaFold
        
        # Create peptide using RDKit
        peptide = Chem.MolFromSequence(peptide_seq)
        if not peptide:
            return None
            
        # Add hydrogens
        peptide = Chem.AddHs(peptide)
        
        # Generate 3D coordinates
        AllChem.EmbedMolecule(peptide)
        
        # Energy minimize
        AllChem.MMFFOptimizeMolecule(peptide)
        
        # Save to file
        output_path = os.path.join(output_dir, "modified_peptide.pdb")
        Chem.MolToPDBFile(peptide, output_path)
        
        return output_path

def main():
    parser = argparse.ArgumentParser(description="PepINVENT Structure-Based Workflow")
    parser.add_argument("--pdb", required=True, help="PDB ID to analyze")
    parser.add_argument("--output", default="./pepinvent_output", help="Output directory")
    parser.add_argument("--sub", nargs="+", help="Substitutions in format POS:AA (e.g., 3:A 5:W)")
    parser.add_argument("--ins", nargs="+", help="Insertions in format POS:AA (e.g., 0:M 4:P)")
    parser.add_argument("--del", nargs="+", help="Deletions as positions (e.g., 2 7)")
    
    args = parser.parse_args()
    
    # Process modifications
    modifications = {"substitutions": {}, "insertions": {}, "deletions": []}
    
    if args.sub:
        for sub in args.sub:
            pos, aa = sub.split(":")
            modifications["substitutions"][int(pos)] = aa
    
    if args.ins:
        for ins in args.ins:
            pos, aa = ins.split(":")
            modifications["insertions"][int(pos)] = aa
    
    if args.del:
        modifications["deletions"] = [int(pos) for pos in args.del]
    
    # Run workflow
    workflow = PepINVENTStructureWorkflow()
    output_dir = os.path.join(args.output, args.pdb)
    
    results = workflow.modify_peptide(args.pdb, modifications, output_dir)
    
    # Print summary
    if 'error' not in results:
        orig = results["original_peptide"]
        mod = results["modified_peptide"]
        
        print("\n=== RESULTS SUMMARY ===")
        print(f"Original peptide: {orig['sequence']}")
        print(f"Modified peptide: {mod['sequence']}")
        print(f"Binding affinity change: {orig['binding_affinity']['binding_affinity']} → {mod['binding_affinity']['binding_affinity']} kcal/mol")
        print(f"Cell penetration probability: {orig['cell_penetration']['cpp_probability']:.2f} → {mod['cell_penetration']['cpp_probability']:.2f}")
        print(f"Structural RMSD: {results['structure_comparison']['rmsd']:.2f} Å")
        print(f"\nDetailed results saved to: {output_dir}/analysis_results.json")
    else:
        print(f"Error: {results['error']}")

if __name__ == "__main__":
    main()