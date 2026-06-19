# Vendored braids-project dependency

This directory contains the unmodified `peyl` Python package used by the
paper-reservoir backbone.

- Upstream: https://github.com/geordw/braids-project
- Upstream commit: `872c63ae6e9a29ed3bb725757d9d38fb0393c85c`
- License: MIT; see `LICENSE`
- Vendored on: 2026-06-14

The annealed-reservoir experiment runs this package in a separate Python
process because the research repository also contains a different local
package named `peyl`. The files under `peyl/` remain unmodified; annealed
selection is implemented outside the vendored dependency.
