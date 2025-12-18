import numpy as np
import pandas as pd
import peptides
from Bio.SeqUtils.ProtParam import ProteinAnalysis


def shannon_entropy(sequence):
    freq = pd.Series(list(sequence)).value_counts(normalize=True)
    return -np.sum(freq * np.log2(freq))

def hydrophobic_moment(sequence):
    analysis = peptides.Peptide(sequence)
    return analysis.hydrophobic_moment(angle=100)


def boman_ind(sequence):
    analysis = peptides.Peptide(sequence)
    return analysis.boman()

def ms_whim(sequence):
    analysis = peptides.Peptide(sequence)
    return analysis.mz()

def calculate_physchem(sequence):
    analysis = ProteinAnalysis(sequence)
    features = {
            'molecular_weight': analysis.molecular_weight(),
            'aromaticity': analysis.aromaticity(),
            'ss_helix': analysis.secondary_structure_fraction()[0],
            'ss_turn': analysis.secondary_structure_fraction()[1],
            'ss_sheet': analysis.secondary_structure_fraction()[2],
            'isoelectric_point': analysis.isoelectric_point(),
            'gravy': analysis.gravy(),
            'molar_extinction_reduced': analysis.molar_extinction_coefficient()[0],
            'instability_index': analysis.instability_index(),
            'shannon_entropy': shannon_entropy(sequence),
            'hydrophobic_moment': hydrophobic_moment(sequence),
            'boman_ind': boman_ind(sequence),
            'ms_whim': ms_whim(sequence),
    }
    return features

