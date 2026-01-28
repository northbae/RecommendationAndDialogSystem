import sys
from pathlib import Path
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset import NewsDataset
from src.features.similarity_metrics import SimilarityCalculator
from src.utils.config import config


def main():

    dataset = NewsDataset()
    df = dataset.load()
    df = dataset.preprocess()

    calculator = SimilarityCalculator(df)

    start_time = time.time()

    calculator.compute_euclidean_similarity()

    calculator.compute_manhattan_similarity()

    calculator.compute_cosine_similarity()

    calculator.compute_chebyshev_similarity()

    calculator.compute_minkowski_similarity(p=3)

    calculator.compute_correlation_similarity()

    calculator.compute_binary_jaccard_similarity()

    calculator.compute_binary_dice_similarity()

    calculator.compute_binary_hamming_similarity()

    calculator.compute_binary_smc_similarity()

    calculator.compute_tree_similarity()

    calculator.compute_tags_similarity()

    calculator.compute_geographic_similarity()

    calculator.compute_author_similarity()

    calculator.compute_time_similarity()

    calculator.compute_comprehensive_similarity()

    elapsed_time = time.time() - start_time

    output_dir = config.get('data.similarity_matrices', 'data/processed/similarity_matrices/')
    calculator.save_all(output_dir)


    for name in calculator.list_available_matrices():
        matrix = calculator.get_similarity_matrix(name)
        mean_val = matrix.values[matrix.values != 1.0].mean()

if __name__ == "__main__":
    main()