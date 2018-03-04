import numpy as np
import pandas as pd
import copy

class Seqlib:
    
    def __init__(self, ninds, nsites):
        self.ninds = ninds
        self.nsites = nsites
        self.seqs = self._simulate()
        self.maf = self._maf()
        
    
    # Make mutated base, later used in function simulate
    def _mutate(self, base):
        diff = set("ACTG") - set(base)
        return np.random.choice(list(diff))
    
    # Simulate a random sequence as arrays of multiple individuals
    def _simulate(self):
        oseq = np.random.choice(list("ACGT"), size=self.nsites)
        arr = np.array([oseq for i in range(self.ninds)])
        muts = np.random.binomial(1, 0.1, (self.ninds, self.nsites))
    
        for col in range(self.nsites):
            newbase = self._mutate(arr[0, col])
            mask = muts[:, col].astype(bool)
            arr[:, col][mask] = newbase
    
        missing = np.random.binomial(1, 0.1, (self.ninds, self.nsites))
        arr[missing.astype(bool)] = "N"    
        return arr
    
    # Return MAF as floats
    # Important to exclude "N" from being counted
    # Create a for-loop structure to compare a given column of bases with the first non-N base
    def _maf(self):
        
        # initiate an array and later iterate over columns
        maf = np.zeros(self.nsites)
        
        for col in range(self.nsites):
            thiscol = self.seqs[:, col]
            nmask = thiscol != "N"
            no_n_len = np.sum(nmask)
            first_non_n_base = thiscol[nmask][0]
            
            freqs = np.sum(thiscol[nmask] != first_non_n_base) / no_n_len
            if freqs > 0.5:
                maf[col] = 1 - freqs
            else:
                maf[col] = freqs
        return maf
    
    # Filter out sequences with missing frequency more than specified max frequency
    # Instead of returning the sequence, return a boolean filter True for columns with Ns > maxmissing
    def _filter_missing(self, maxmissing):
        freqmissing = np.sum(self.seqs == "N", axis=0) / self.seqs.shape[0]
        return freqmissing > maxmissing
    
    # Filter out sequences with minor allel sequence frequency greater than the specified frequency
    # Instead of returning the sequence, return a boolean filter True for columns with maf < minmaf
    def _filter_maf(self, minmaf):
        return self.maf < minmaf
    
    # Apply maf and missing filters to the array
    def filter(self, maxmissing, minmaf):
        filter1 = self._filter_maf(minmaf)
        filter2 = self._filter_missing(maxmissing)
        fullfilter = filter1 + filter2 # Q: if one is "True" and the other is not, then sum is "False"?
        return self.seqs[:, np.invert(fullfilter)]
    
    # Apply maf and missing filters
    # Return a copy of seqlib object where the .seqs array has been filtered
    def filter_seqlib(self, minmaf, maxmissing):
        newseqs = self.filter(minmaf, maxmissing)
        newself = copy.deepcopy(self) # Q: what does this do?
        newself.__init__(newseqs.shape[0], newseqs.shape[1])
        
        newself.seqs = newseqs # overwrite it
        newself._maf() # why this?
        return newself
        
    
    def calculate_statistics(self):
        nd = np.var(self.seqs == self.seqs[0], axis=0).mean()
        mf = np.mean(np.sum(self.seqs != self.seqs[0], axis=0) / self.seqs.shape[0])
        inv = np.all(self.seqs == self.seqs[0], axis=0).sum()
        var = self.seqs.shape[1] - inv
    
        # return all values as panda series with specified name
        return pd.Series(
            {"mean nucleotide diversity": nd,
             "mean minor allele frequency": mf,
             "invariant sites": inv,
             "variable sites": var,
            })