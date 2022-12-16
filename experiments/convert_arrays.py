import sys
from pyreadr import read_r
import torch


results = read_r(sys.argv[1])
results = [torch.from_numpy(result.to_numpy()) for result in results.values()]
for result in results:
    print(result)
torch.save(results, sys.argv[2])