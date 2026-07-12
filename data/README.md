# Data policy

The benchmark uses a deterministic, read-only subset of the public
ColO-RAN dataset. The upstream dataset is GPL-3.0 and is not redistributed in
this repository. Run:

```bash
python scripts/fetch_colosseum_subset.py
```

The downloader pins upstream commit
`bd86629d07d5fbfb778ebe3afd9d0b05e5191c6b`, verifies every downloaded file,
and writes `data/colosseum_subset_manifest.json`. Raw and processed traces are
ignored by Git. The subset spans all three schedulers, three resource-allocation
configurations, two repetitions, two base stations, and three UEs per base
station---one UE per traffic slice (108 UE traces).

Upstream dataset: <https://github.com/wineslab/colosseum-oran-coloran-dataset>

Required citation: M. Polese *et al.*, "ColO-RAN: Developing Machine
Learning-Based xApps for Open RAN Closed-Loop Control on Programmable
Experimental Platforms," *IEEE Transactions on Mobile Computing*, vol. 22,
no. 10, pp. 5787-5800, 2023, DOI:
<https://doi.org/10.1109/TMC.2022.3188013>.
