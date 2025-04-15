import os
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import tempfile
import subprocess
import warnings

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import biotite.sequence as seq
import biotite.application.dssp as dssp
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.DSSP import DSSP

#python structure_workflow.py --pdb 1MFG --sub 2:A 5:W --ins 0:G --del 7

class PepStructureAnalyzer:
    """Analyze peptide structures and their binding interactions."""
    
    def __init__(self, pdb_cache_dir=None):
        """Initialize the analyzer with optional PDB cache directory."""
        self.pdb_cache_dir = pdb_cache_dir or os.path.join(os.path.expanduser("~"), ".pepinvent_pdb_cache")
        os.makedirs(self.pdb_cache_dir, exist_ok=True)
        
        self.pdb_parser = PDBParser(QUIET=True)
        self.pdb_io = PDBIO()
        
    def fetch_pdb_structure(self, pdb_id: str) -> str:
        """Fetch PDB structure and return local file path."""
        pdb_id = pdb_id.lower()
        local_path = os.path.join(self.pdb_cache_dir, f"{pdb_id}.pdb")
        
        if not os.path.exists(local_path):
            print(f"Downloading PDB {pdb_id}...")
            url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            try:
                import urllib.request
                urllib.request.urlretrieve(url, local_path)
            except Exception as e:
                print(f"Error downloading {pdb_id}: {e}")
                return None
                
        return local_path
    
    def extract_peptide_from_structure(self, pdb_path: str, chain_id=None) -> Tuple[str, Dict]:
        """Extract peptide sequence and data from PDB structure."""
        structure = self.pdb_parser.get_structure("peptide", pdb_path)
        
        # If chain_id not specified, try to find peptide chains
        chains = list(structure[0].get_chains())
        if not chain_id:
            # Heuristic: peptide chains are usually shorter
            chains_by_length = sorted(chains, key=lambda c: len(list(c.get_residues())))
            chain = chains_by_length[0]  # Shortest chain
        else:
            chain = next((c for c in chains if c.id == chain_id), None)
            if not chain:
                print(f"Chain {chain_id} not found in structure")
                return None, {}
        
        # Extract sequence
        residues = list(chain.get_residues())
        sequence = ""
        for res in residues:
            if res.get_id()[0] == ' ':  # Standard residue
                sequence += seq.ProteinSequence.convert_letter_3to1(res.get_resname())
        
        # Extract properties
        properties = {
            "length": len(sequence),
            "n_chains": len(list(structure[0].get_chains())),
            "extracted_chain": chain.id
        }
        
        return sequence, properties
    
    def peptide_to_smiles(self, peptide_seq: str) -> str:
        """Convert peptide sequence to SMILES representation."""
        # This is a simplified conversion - would need to be expanded
        aa_to_smiles = {
            "A": "N[C@@H](C)C(=O)",
            "R": "N[C@@H](CCCNC(=N)N)C(=O)",
            # Add other amino acids here
        }
        
        smiles = ""
        for aa in peptide_seq:
            if aa in aa_to_smiles:
                smiles += aa_to_smiles[aa]
            else:
                smiles += "?"  # Unknown amino acid
        
        return smiles
    
    def estimate_binding_affinity(self, receptor_path: str, ligand_path: str) -> Dict:
        """Estimate binding affinity using AutoDock Vina."""
        try:
            # This requires AutoDock Vina to be installed
            with tempfile.NamedTemporaryFile(suffix='.pdbqt') as receptor_pdbqt, \
                 tempfile.NamedTemporaryFile(suffix='.pdbqt') as ligand_pdbqt, \
                 tempfile.NamedTemporaryFile(suffix='.pdbqt') as out_pdbqt:
                
                # Convert PDB to PDBQT
                subprocess.run(f"obabel {receptor_path} -O {receptor_pdbqt.name}", shell=True, check=True)
                subprocess.run(f"obabel {ligand_path} -O {ligand_pdbqt.name}", shell=True, check=True)
                
                # Run Vina
                vina_cmd = f"vina --receptor {receptor_pdbqt.name} --ligand {ligand_pdbqt.name} --out {out_pdbqt.name} --exhaustiveness 8"
                result = subprocess.run(vina_cmd, shell=True, capture_output=True, text=True, check=True)
                
                # Parse the output to get binding affinity
                affinity = None
                for line in result.stdout.split('\n'):
                    if "Affinity:" in line:
                        affinity = float(line.split()[1])
                        break
                
                return {"binding_affinity": affinity, "unit": "kcal/mol"}
        except Exception as e:
            print(f"Error estimating binding affinity: {e}")
            return {"binding_affinity": None, "error": str(e)}
    
    def predict_cell_penetration(self, peptide_seq: str) -> Dict:
        """Predict cell-penetrating potential of peptide."""
        # Simplified model based on basic properties
        # In practice, you'd use a more sophisticated ML model or tool
        
        # Simple heuristics:
        n_res = len(peptide_seq)
        n_pos = peptide_seq.count('R') + peptide_seq.count('K') + peptide_seq.count('H')
        n_neg = peptide_seq.count('D') + peptide_seq.count('E')
        n_aromatic = peptide_seq.count('F') + peptide_seq.count('W') + peptide_seq.count('Y')
        net_charge = n_pos - n_neg
        
        # Simplified scoring function based on CPP characteristics
        cpp_score = (0.5 * net_charge) + (0.2 * n_aromatic) - (0.1 * n_res)
        
        # Normalize to a probability-like scale
        probability = min(max(0, (cpp_score + 2) / 5), 1)
        
        return {
            "cpp_probability": probability,
            "net_charge": net_charge,
            "length": n_res,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "n_aromatic": n_aromatic
        }
    
    def compare_structures(self, original_pdb: str, modified_pdb: str) -> Dict:
        """Compare two protein structures and quantify differences."""
        try:
            # Load structures using biotite
            original_array = struc.array(pdb.PDBFile.read(original_pdb).get_structure())
            modified_array = struc.array(pdb.PDBFile.read(modified_pdb).get_structure())
            
            # Calculate RMSD for CA atoms
            original_ca = original_array[original_array.atom_name == "CA"]
            modified_ca = modified_array[modified_array.atom_name == "CA"]
            
            # If different number of CA atoms, align what's possible
            min_length = min(len(original_ca), len(modified_ca))
            rmsd = struc.rmsd(original_ca[:min_length], modified_ca[:min_length])
            
            # Secondary structure comparison using DSSP
            try:
                original_ss = dssp.DsspApp(original_pdb).get_secondary_structure()
                modified_ss = dssp.DsspApp(modified_pdb).get_secondary_structure()
                
                ss_similarity = sum(1 for a, b in zip(original_ss, modified_ss) if a == b) / len(original_ss)
            except Exception:
                ss_similarity = None
                
            return {
                "rmsd": rmsd,
                "ca_atoms_compared": min_length,
                "secondary_structure_similarity": ss_similarity
            }
        except Exception as e:
            print(f"Error comparing structures: {e}")
            return {"error": str(e)}
    
    def analyze_peptide_pdb(self, pdb_id: str, chain_id=None) -> Dict:
        """Full analysis of a peptide in PDB."""
        pdb_path = self.fetch_pdb_structure(pdb_id)
        if not pdb_path:
            return {"error": f"Could not fetch PDB {pdb_id}"}
        
        # Extract peptide sequence and properties
        peptide_seq, properties = self.extract_peptide_from_structure(pdb_path, chain_id)
        if not peptide_seq:
            return {"error": f"Could not extract peptide from PDB {pdb_id}"}
        
        # Convert to SMILES for PepINVENT
        smiles = self.peptide_to_smiles(peptide_seq)
        
        # Cell penetration prediction
        cpp_prediction = self.predict_cell_penetration(peptide_seq)
        
        # Combine results
        results = {
            "pdb_id": pdb_id,
            "peptide_sequence": peptide_seq,
            "smiles": smiles,
            "properties": properties,
            "cell_penetration": cpp_prediction,
        }
        
        return results

def extract_peptide_binding_pairs(pdb_id: str, output_dir: str) -> Dict:
    """Extract peptide-protein binding pairs from PDB and save to separate files."""
    analyzer = PepStructureAnalyzer()
    pdb_path = analyzer.fetch_pdb_structure(pdb_id)
    if not pdb_path:
        return {"error": f"Could not fetch PDB {pdb_id}"}
    
    structure = analyzer.pdb_parser.get_structure(pdb_id, pdb_path)
    
    # Identify peptide chains (typically shorter)
    chains = list(structure[0].get_chains())
    chain_lengths = {chain.id: len(list(chain.get_residues())) for chain in chains}
    
    # Simple heuristic: the shortest chain is likely the peptide
    chains_sorted = sorted(chain_lengths.items(), key=lambda x: x[1])
    
    if len(chains_sorted) < 2:
        return {"error": "Structure contains only one chain"}
    
    peptide_chain_id = chains_sorted[0][0]
    receptor_chain_ids = [chain_id for chain_id, _ in chains_sorted[1:]]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract peptide
    class PeptideSelect(Select):
        def accept_chain(self, chain):
            return chain.id == peptide_chain_id
    
    peptide_path = os.path.join(output_dir, f"{pdb_id}_peptide.pdb")
    analyzer.pdb_io.set_structure(structure)
    analyzer.pdb_io.save(peptide_path, PeptideSelect())
    
    # Extract receptor
    class ReceptorSelect(Select):
        def accept_chain(self, chain):
            return chain.id in receptor_chain_ids
    
    receptor_path = os.path.join(output_dir, f"{pdb_id}_receptor.pdb")
    analyzer.pdb_io.set_structure(structure)
    analyzer.pdb_io.save(receptor_path, ReceptorSelect())
    
    # Extract peptide sequence
    peptide_seq, properties = analyzer.extract_peptide_from_structure(peptide_path)
    
    return {
        "pdb_id": pdb_id,
        "peptide_chain": peptide_chain_id,
        "receptor_chains": receptor_chain_ids,
        "peptide_path": peptide_path,
        "receptor_path": receptor_path,
        "peptide_sequence": peptide_seq,
        "peptide_length": properties["length"]
    }

def main():
    """Example usage of the PepStructureAnalyzer."""
    # Example PDB IDs with peptide-protein complexes
    example_pdbs = ["1MFG", "2FTL", "1YY9"]
    
    analyzer = PepStructureAnalyzer()
    
    for pdb_id in example_pdbs:
        print(f"\nAnalyzing PDB: {pdb_id}")
        results = analyzer.analyze_peptide_pdb(pdb_id)
        print(f"Peptide sequence: {results.get('peptide_sequence', 'Not found')}")
        print(f"Cell penetration probability: {results.get('cell_penetration', {}).get('cpp_probability', 'N/A'):.2f}")
        
        # Extract binding pairs
        output_dir = f"./pdb_extracts/{pdb_id}"
        binding_pair = extract_peptide_binding_pairs(pdb_id, output_dir)
        print(f"Extracted files to {output_dir}")
        
        if 'error' not in binding_pair:
            # Estimate binding affinity
            affinity = analyzer.estimate_binding_affinity(
                binding_pair["receptor_path"], 
                binding_pair["peptide_path"]
            )
            print(f"Estimated binding affinity: {affinity.get('binding_affinity', 'N/A')} {affinity.get('unit', '')}")

if __name__ == "__main__":
    main()