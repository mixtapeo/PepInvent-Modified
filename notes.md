- Evaluation metrics
    - long ver
        1. Validity: A generated peptide with a syntactically accurate SMILES that follows the chemical
        rules such as valency and chemical bonding was categorized as valid, with validity assessed
        using RDKit.
        2. Uniqueness: It is defined in multiple levels:
        2.1. Peptide-level uniqueness: The number of unique SMILES strings after the separators are
        removed and the generated peptide is canonicalized with chirality. As the model generates
        amino acids to complete an input peptide, the generation of two peptides from the same
        input might contain the same amino acids in different orders. This makes the two peptides
        unique but the unique set of amino acids to be duplicated.
        2.2. Amino acid-level uniqueness: This was evaluated in three levels to detect the non
        canonical, stereochemical and canonical variability of the generated building blocks,
        respectively to:
        a) String-Level Uniqueness refers to the number of amino acids strings generated
        being unique by comparing them character by character.
        b) Isomeric SMILES-Level Uniqueness, similarly to the peptide-level uniqueness, is
        the number of unique amino acids after the SMILES strings are canonicalized while
        retaining the chirality.
        c) Canonical SMILES-Level Uniqueness is the unique amino acids with
        canonicalization as the molecules stripped off their stereochemical information. This
        offers the standardized representation where the uniqueness is ensured by a distinct
        molecular structure.
        3. Novelty: The novelty was calculated by profiling the unique generated amino acids as natural,
        non-natural and novel. In this case, non-natural refers to the NNAAs utilized to create the
        semi-synthetic peptide data for model training whereas novel is the NNAAs that are
        generated by the model that do not exist in the training set.
    - Validity: RDKit
    - 
- Solubility
    - Researchers used CAMSOL-PTM solubility scorer algorithm: [Sequence-based prediction of the intrinsic solubility of peptides containing non-natural amino acids | Nature Communications](https://www.nature.com/articles/s41467-023-42940-w)
        - for peptides composed of natural amino acids or NNAAs.
        - Citrulline: ADFFKLFDE|C(C[C@@H](notion://www.notion.so/mixtapeo7/C(=O)O)N)CNC(=O)N|YPLKDDSEDR:
            - @ pH 7:
            
            ![image.png](attachment:35846554-d10b-410f-a35f-523378addcf9:image.png)
            
            - “The protein variant intrinsic solubility score is 1.410045”
        - CAMSOL-PTM was integrated to RL framework as a scoring component where the input molecule is the generated peptide → MIGHT BE GOOD TO EXPLORE RL AS WELL
- Steps to run pepinvent:
    1. your_sampling_parameters.json:
        
        ```json
        {
            "sampling_rate": 0.05,      // Controls diversity of sampling
            "temperature": 298,         // Higher values = more diverse outputs
            "num_samples": 10,          // Number of peptides to generate per input
            "batch_size": 32,           // Number of sequences to process together
            "seed": 42,                 // For reproducible results
            "algorithm": "monte_carlo",
            "convergence_threshold": 0.001,
            "model_type": "mol2mol",
            "model_path": "./models/generative_model.ckpt",
            "results_output": "./results/sampling_output.json",
            "input_sequences_path": "./input_sequences.csv"
        }
        
        ```
        
    2. python input_to_sampling.py your_sampling_parameters.json
        - Sampling Parameters Explained
            - **sampling_rate**: `0.05`
                - Controls the exploration rate during sampling.
                - Lower values = more focused sampling; higher values = more exploration.
                - **temperature**: `298`
                    - Controls randomness in the generation process (typically 298K = room temperature).
                    - Higher values produce more diverse outputs, lower values make sampling more deterministic.
            - **seed**: `42`
                - Random seed for reproducibility.
                - Using the same seed ensures consistent results between runs.
            - **algorithm**: `"monte_carlo"`
                - Sampling algorithm used for molecule generation. Monte Carlo methods use random sampling for sequence exploration.
            - Execution Controls
                - **max_iterations**: `500`
                    - Maximum number of iterations the sampling algorithm will run.
                    - Higher values allow more thorough exploration but take longer.
                    - **num_samples**: `10` (Note: this appears to have a typo, should be `num_samples`)
                        - Number of different peptides to generate per input sequence.
                        - Increase for more diverse outputs.
                    - **convergence_threshold**: `0.001`
                        - Threshold at which the algorithm is considered converged.
                        - If changes between iterations fall below this value, sampling might stop early.
- How to Validate Peptides
    
    ### **Validating Generated Peptides in PepINVENT**
    
    Based on the paper, PepINVENT validates generated peptides through several methods:
    
    1. **RDKit-Based Chemical Validation:**
        - A generated peptide with a syntactically accurate SMILES that follows the chemical rules such as valency and chemical bonding is categorized as valid, with validity assessed using RDKit.
    2. **Multiple Uniqueness Checks:**
        - Peptide-level uniqueness (canonical SMILES)
        - Amino acid-level uniqueness at three levels: string, isomeric SMILES, and canonical SMILES.
    3. **Topology Validation:**
        - The paper describes testing if generated peptides maintain correct topological arrangements (e.g., for cyclic peptides, ensuring proper ring formation).
    4. **Novelty Assessment:**
        - Classifying generated amino acids as natural, known non-natural, or novel non-natural.
- `validator.py` Explanation
    
    `validator.py` is a comprehensive validation tool for analyzing peptide molecules generated by the PepINVENT framework. Here's what it does and how it works:
    
    ### Main Purpose
    
    The script evaluates generated peptides (represented as SMILES strings) on several key dimensions:
    
    1. **Validity**: Checks if peptide structures are chemically valid.
    2. **Uniqueness**: Measures diversity at both peptide and amino acid levels.
    3. **Novelty**: Identifies novel amino acids compared to natural and training sets.
    4. **Topology**: Classifies structures as linear, macrocyclic, or other cyclic peptides.
    
    Key Components and Functions
    
    ### PeptideValidator Class
    
    - **`__init__` and `_get_default_natural_aa`**: Initialize with reference sets of natural and non-natural amino acids.
    - **`validate_peptides`**: Core function that processes each peptide through multiple validation steps:
        - Converts SMILES to RDKit molecules to verify basic chemical validity.
        - Checks ring structures using `GetRingInfo().AtomRings()` to classify topology.
        - Identifies amide bonds using `_is_amide_bond()` to verify peptide characteristics.
        - Splits peptides by '\' separator to analyze individual amino acids.
        - Tracks statistics in the results dictionary.
    - **`_is_amide_bond`**: Helper that detects peptide bonds by checking for C-N single bonds where C is part of a carbonyl.
    
    ### run_validation Function
    
    - Loads peptide SMILES from CSV files (typically from model sampling output).
    - Creates a validator instance and runs validation.
    - Prints detailed statistics and saves results to JSON.
    
    ### Why It Calls Certain Functions
    
    - **RDKit functions**: These provide chemical structure analysis capabilities:
        - `MolFromSmiles`: Converts SMILES to manipulable molecule objects.
        - `MolToSmiles`: Standardizes structures for comparison.
        - `GetRingInfo()`: Analyzes ring systems for topology classification.
    - **Exception handling**: Wraps critical operations to prevent crashes on invalid molecules.
    - **`list(results['novelty'].keys())`**: Creates a static copy of keys before modifying the dictionary to prevent the "dictionary changed size during iteration" error.
- Workflow
    1. Loads peptide data from a CSV file.
    2. Validates each peptide through multiple checks.
    3. Calculates statistics on validity, uniqueness, novelty, and topology.
    4. Generates a detailed report with percentages and counts.
    5. Saves all information to a JSON file for further analysis.
    
    This validator plays a crucial role in evaluating the quality of the generative model's output, providing metrics that help assess whether the generated peptides meet chemical and structural requirements while exhibiting desired diversity characteristics.
    

which database did exploring the potential local machine learning in comparison to the
how did they choose NNAA, how validate
validate results of autodock vina, (binding affinity) (sanity checks)
and permiability.

- report:
    
    Solubility:
    
    - Camsol PTM is integrated into RL framework as a scoring param
    
    permiability:
    
    - predict_proba() from predictive_model.py: embedded into model during training. Very accurate.
    - Uniqueness:
        - Show the excerpt from paper
        - 2.1 Two peptides are considered **unique** if their full sequences are different, even if they are composed of the same amino acids in different orders. ORDER MATTERS; PERMUTATIONS OKAY
        - a) String-Level Uniqueness → Checks uniqueness by comparing amino acid sequences character by character (exact string match).
            
            b) Isomeric SMILES-Level Uniqueness → Converts sequences into SMILES format, keeping chirality (stereochemistry), then checks for unique structures.
            
            c) Canonical SMILES-Level Uniqueness → Converts to canonical SMILES (standardized structure) but ignores chirality, ensuring uniqueness based only on molecular structure.
            
    - Binding affinity:
        - havent done much yet.
        - CAMP can serve as a useful tool in peptide-protein interaction prediction and identification of important binding residues in the peptides, which can thus facilitate the peptide drug discovery process.
            - https://www.nature.com/articles/s41467-021-25772-4
            - https://www.sciencedirect.com/science/article/pii/S2590123023004620#da0010
- powerpoint expalining pepinvent. whats the input whats the goal. other tools ive found that could be useful for evaluating predictions (permeability, binding affinity, solubility)
- to do:
    - How to optimise the question marks. Does RL find new ?s on its own? can something do that?
        - No, Monte carlo
    - Binding affinity figure out
        - Binding affinity not exposed publicly through scoring function or sampling json.
            - ⇒ Must train our own model: I think we have their data but im not really sure.
        - FlexPepDock
            - **peptide–protein interactions**
                - No code available, but web server available. Is also computationaly expensive and takes 2h for results
                - best option for natural peptide - protein has options for NNAA but am not sure
                    - “currently, the server **supports** the docking of peptides containing modified amino acids (such as phosphorylation, acetylation, etc.) only in high-resolution mode. When submitting such complexes, the user should consult the FAQ page for exact format of the modified residue. “
        - https://pubs.rsc.org/en/content/articlehtml/2021/sc/d1sc02322e
        - add these to custom alerts
    

show examples of how it makes sourec more soluble, permaible, etc.

how can we determine how it docks to protein 

NAA → PEPINVENT → OLD WITH an NNAA → CHECK IF IT DOCKS WITH PROTEIN