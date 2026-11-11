import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from load_data import load_dataset
from exploration import run_exploration


def main():
    print("Running Person 1 Tasks (Setup, Data Acquisition & Exploration)")

    print("\nTask 1: Loading raw Kaggle Twitter corpus")
    load_dataset()

    print("\nTask 2: Running corpus statistics and generating plots")
    run_exploration()

    print("\nPerson 1 tasks finished successfully.")
    print("Figures saved in person_1/figures/")
    print("Notebook ready at person_1/notebooks/01_exploration.ipynb")


if __name__ == "__main__":
    main()
