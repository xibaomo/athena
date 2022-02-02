import numpy as np
import pandas as pd
import sys, os

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python {} <csv_file> <markov.yaml>".format(sys.argv[1]))
        sys.exit()
