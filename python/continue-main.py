from glauber.glauberFixIndices import GlauberSimulatorFixIndices
from glauber.glauberDynIndices import GlauberSimDynIndices
import logging
logging.basicConfig(level=logging.DEBUG)
import json
import os
import concurrent.futures as fts
import multiprocessing as mp
import socket
import sys
import gitinfo  
import argparse
import run_multiple_for_traces

parser=argparse.ArgumentParser()

parser.add_argument("--stem", help="root Dirs of runs to continue")
parser.add_argument("--checkpoint_freq", help="Frequency of checkpoints")
parser.add_argument("--extra_steps", help="how many more steps to do")

def main():
    args=parser.parse_args()
    stem = args.stem
    checkpoint_freq = int(args.checkpoint_freq)
    extra_steps = int(args.extra_steps)

    with open(os.path.join(stem, "run_multiple_for_traces.json"), "r") as f:
        namespace = json.load(f)
    
    namespace["checkpoint"] = checkpoint_freq
    namespace["t"] = namespace["t"] + extra_steps

    args = []
    for key, value in namespace.items():
        if value is not None:
            args.append(json.dumps(key))
            args.append(json.dumps(value))
    
    main = run_multiple_for_traces.Main()