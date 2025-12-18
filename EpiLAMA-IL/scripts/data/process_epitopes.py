import argparse
from pathlib import Path
import pandas as pd

from .utils import filter_mhc_class, filter_by_length, remove_ambiguous_epitopes, filter_cytokine_class


def process_tcell_data(file_path):
    tcell_data = pd.read_csv(file_path)
    tcell_data = filter_mhc_class(tcell_data, column_name='MHC Class', mhc_class='II')
    tcell_data = filter_by_length(tcell_data, column_name='Epitope Seq', min_length=13, max_length=25)

    # Filter rows based on unique labels per epitope and response measured
    filtered_data = remove_ambiguous_epitopes(tcell_data)

    filtered_data = filtered_data.drop(columns=['Evidence Code', 'Antigen Source', 'Starting Position_', 'Ending Position_', 'MHC Class', 'MHC Allele'])
    filtered_data = filtered_data.drop_duplicates()
    filtered_data = filter_cytokine_class(filtered_data, column_name='Response measured', cytokine_class='IFNg release')
    return filtered_data


def main():
    parser = argparse.ArgumentParser(description="Process raw epitope CSV into filtered dataset")
    parser.add_argument("--input", "-i", required=False, default="data/external/tcell_full_v3_processed_extended.csv",
                        help="Path to raw epitope CSV")
    parser.add_argument("--output", "-o", required=False, default="data/processed/base_tcell_data_ifng.csv",
                        help="Path to write processed CSV")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    filtered_tcell_data = process_tcell_data(input_path)
    filtered_tcell_data.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
