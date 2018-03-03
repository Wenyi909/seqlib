import numpy as np
import pandas as pd

class Seqlib:
    
    def __init__(self, ninds, nsites):
        self.ninds = ninds
        self.nsites = nsites
        self.seqs = self.simulate()
        self.maf = self.maf()
        
    
    # Make mutated base, later used in function simulate
    def mutate(self, base):
        diff = set("ACTG") - set(base)
        return np.random.choice(list(diff))
    
    # Simulate a random sequence as arrays of multiple individuals
    def simulate(self):
        oseq = np.random.choice(list("ACGT"), size=self.nsites)
        arr = np.array([oseq for i in range(self.ninds)])
        muts = np.random.binomial(1, 0.1, (self.ninds, self.nsites))
    
        for col in range(self.nsites):
            newbase = self.mutate(arr[0, col])
            mask = muts[:, col].astype(bool)
            arr[:, col][mask] = newbase
    
        missing = np.random.binomial(1, 0.1, (self.ninds, self.nsites))
        arr[missing.astype(bool)] = "N"    
        return arr
    
    # Return MAF as floats
    def maf(self):
        freqs = np.sum(self.seqs != self.seqs[0], axis=0) / self.seqs.shape[0]
        maf = freqs.copy()
        maf[maf > 0.5] = 1 - maf[maf > 0.5]
        return maf
    
    # Filter out sequences with missing frequency more than specified max frequency
    def filter_missing(self, maxmissing):
        freqmissing = np.sum(self.seqs == "N", axis=0) / self.seqs.shape[0]
        return self.seqs[:, freqmissing <= maxmissing]
    
    # Filter out sequences with minor allel sequence frequency greater than the specified frequency
    def filter_maf(self, minmaf):
        freqs = np.sum(self.seqs != self.seqs[0], axis=0) / self.seqs.shape[0]
        maf = freqs.copy()
        maf[maf > 0.5] = 1 - maf[maf > 0.5]
        return self.seqs[:, maf > minmaf]
    
    def filter(self, maxmissing, minmaf):
        freqmissing = np.sum(self.seqs == "N", axis=0) / self.seqs.shape[0]
        arr = self.seqs[:, freqmissing <= maxmissing]
        freqs = np.sum(arr != arr[0], axis=0) / arr.shape[0]
        maf = freqs.copy()
        maf[maf > 0.5] = 1 - maf[maf > 0.5]
        return arr[:, maf > minmaf]
    
    def calculate_statistics(self):
        nd = np.var(self.seqs == self.seqs[0], axis=0).mean()
        mf = np.mean(np.sum(self.seqs != self.seqs[0], axis=0) / self.seqs.shape[0])
        inv = np.any(self.seqs != self.seqs[0], axis=0).sum()
        var = self.seqs.shape[1] - inv
    
        # return all values as panda series with specified name
        return pd.Series(
            {"mean nucleotide diversity": nd,
             "mean minor allele frequency": mf,
             "invariant sites": inv,
             "variable sites": var,
            })

