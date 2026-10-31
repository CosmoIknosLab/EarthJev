# Data scripts

Download and preparation scripts for each dataset.

- Data go under `$S1GEO_DATA` (default `./data`, not versioned), never in `$HOME`.
- Check the license and size on the official page of each dataset before downloading.
- Data are stored as compressed Zarr (one shard file per 256-512 patches), not as millions of small files.
