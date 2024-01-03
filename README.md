## pySSA

pySSA is a Python package to compute the singular spectrum analysis of multiple time series. pySSA supports multiple time series at once, and parallel computation as well. Possible return values for computation can be:

* `singular_values`: Returns the singular values associated with the SSA reconstruction
* `reconstruction`: Returns the reconstructed time series to the desired rank
* `full`: Returns both the above as well as the rank-components associated with each singular value. These can be summed to provide the reconstruction of a given rank (indeed, that is what happens under-the-hood when this option is set). 

You can also use either a full SVD or a randomised SVD, based on the algorithm by Halko et. al (2011).

## Usage

pySSA expects a `TimeSeries` object to have been initialised containing the variables `data` (a M * N numpy array, containing M time series of length N), `window_size`, and the `rank`. Next, we set a `Config` object containing the `svd_method` (`full` or `randomized`), the `return_data` (discussed beforehand) and a boolean parallel computation flag. 

To use pySSA, then initialise the `SSA` class with the time series and config set. Next, activate the `get_hankel_matrices()` method, and lastly the `compute_ssa()` method. Following this last step, the appropriate instance attributes (`SSA.singular_values, SSA.ssa_reconstructions, SSA.components` are set). 

See below for an example in action.

## Example Usage
```
series = np.random.rand(3, 10)
window_size = 3
rank = 2
time_series = TimeSeries(data=series, window_size=window_size, rank=rank)
config = Config(svd_method="full", return_data="singular_values", parallel=False)

ssa = SSA(time_series, config)
hankel_matrices = ssa.get_hankel_matrices()
ssa.compute_ssa()

print(ssa.singular_values)
```

Inspiration for this project comes from the research @zellaa did for her dissertation. Namely showing that distance metrics can be a sufficient substitute for 'similarity' when comparing de-noised time series data. 

## TODO
- Add to PyPI
- Speed up computations with Numba
- Flushout README
