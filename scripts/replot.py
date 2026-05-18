import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.plot_results import replot_saved_results


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Replot a saved experiment .pkl/.json result payload."
    )
    parser.add_argument("results_path", help="Path to saved results payload.")
    args = parser.parse_args(argv)
    replot_saved_results(args.results_path)


if __name__ == "__main__":
    main()
