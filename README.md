# adsorb-screen

Automated, general-purpose **AdsorbML** configuration screening of molecules, radicals, and atoms on **any catalytic surface** using FAIRChem v2 / UMA and VASP input generation.

## Key Features
- **Any Adsorbate**: Screen built-in radicals (`CH2I2`, `CH2I`, `CH2`, `CH2OO`), any single atom (`H`, `O`, `N`, `I`, `Pt`), any molecule in the ASE G2 database (`CO`, `CO2`, `NH3`, `CH3OH`, `H2O`, `C2H4`), SMILES strings (`smiles:CC(=O)O`), or custom structure files (`.xyz`, `POSCAR`, `.sdf`).
- **Any Catalytic Surface**: Provide your own slab (`--slab-file`), cleave any bulk crystal file (`--bulk my_oxide.cif --millers 1 1 0`), use built-in oxide presets (`alpha-quartz`, `rutile-TiO2`, `MgO`), use any ASE metal/crystal (`--bulk Pt --millers 1 1 1`), or fetch directly from the Materials Project (`--bulk mp-126 --millers 1 1 1`).
- **Symmetry-Aware Binding Modes**: Automatically identifies symmetry-unique monodentate and bidentate binding sites and orients lone pairs/radical lobes toward the surface (`down`), pi-systems parallel to the surface (`flat`), or bidentate bridges (`pair`).
- **Anomaly Filtering & VASP Export**: Rejects dissociated, desorbed, intercalated, or reconstructed trajectories and exports ready-to-run VASP directories (`POSCAR`, `INCAR`, `KPOINTS`, `POTCAR_ORDER`) with open-shell `MAGMOM`/`NUPDOWN` and metallic/insulator smearing.

## Installation

```bash
pip install -e .
# Optional extras for SMILES and Materials Project support:
pip install -e ".[smiles,mp,dev]"
huggingface-cli login
```
