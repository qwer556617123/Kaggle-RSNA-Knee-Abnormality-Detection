# RSNA Knee research toolkit

Offline Python utilities for MRI preprocessing, study-level modeling, training,
fold handling, and prediction validation developed during the RSNA knee competition.
No medical data, model weights, credentials, or cloud automation are distributed.

## Installation

Use Python 3.10 or later in a separate environment:

```sh
python -m pip install -e .
```

Install the optional `models` dependencies when using neural backbone interfaces.
Only load checkpoints from trusted sources. This code is not clinically validated.
