from __future__ import annotations

from pathlib import Path
import numpy as np
from ase import Atoms
from ase.io import write


INCAR_COMMON = {
    "PREC": "Normal",
    "ENCUT": 400,
    "GGA": "RP",
    "IVDW": 11,
    "ISPIN": 1,
    "EDIFF": 1e-5,
    "ISMEAR": 0,
    "SIGMA": 0.05,
    "LREAL": "Auto",
    "ISYM": 0,
    "NELM": 200,
    "LWAVE": ".FALSE.",
    "LCHARG": ".FALSE.",
    "IBRION": 2,
    "NSW": 500,
    "ISIF": 0,
    "EDIFFG": -0.03,
}
INCAR_SLAB = {"LDIPOL": ".TRUE.", "IDIPOL": 3}
INCAR_GAS = {"LDIPOL": ".TRUE.", "IDIPOL": 4}


def write_vasp_dir(
    atoms: Atoms,
    outdir: Path,
    incar_extra: dict,
    kpts: tuple[int, int, int] | None = None,
    unpaired: int = 0,
    fix_moment: bool = False,
    metallic_slab: bool = False,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # Sort explicitly by symbol so tags stay synchronized with POSCAR atom order
    symbols_orig = atoms.get_chemical_symbols()
    sort_idx = np.argsort(symbols_orig, kind="mergesort")
    sorted_atoms = atoms[sort_idx]

    poscar_path = outdir / "POSCAR"
    write(str(poscar_path), sorted_atoms, format="vasp", direct=True, sort=False)

    symbols = sorted_atoms.get_chemical_symbols()
    species = list(dict.fromkeys(symbols))

    incar = {**INCAR_COMMON, **incar_extra}
    if metallic_slab and incar_extra != INCAR_GAS:
        incar["ISMEAR"] = 1
        incar["SIGMA"] = 0.20

    if unpaired > 0:
        incar["ISPIN"] = 2
        if fix_moment:
            incar["NUPDOWN"] = unpaired
        tags = sorted_atoms.get_tags()
        # Assign initial spin to non-H adsorbate atoms (tag == 2) or isolated gas molecule
        is_gas = (tags == 2).sum() == 0
        is_mag_target = [
            ((t == 2 or is_gas) and (s != "H" or len(sorted_atoms) == 1))
            for t, s in zip(tags, symbols)
        ]
        n_mag = max(sum(is_mag_target), 1)
        mag_val = round(max(unpaired / n_mag, 0.5), 2)
        incar["MAGMOM"] = " ".join(f"{mag_val}" if m else "0.0" for m in is_mag_target)

    with open(outdir / "INCAR", "w") as f:
        f.write("".join(f"{k} = {v}\n" for k, v in incar.items()))

    if kpts is None:
        a, b = sorted_atoms.cell.lengths()[:2]
        kpts = (max(1, round(40.0 / a)), max(1, round(40.0 / b)), 1)

    with open(outdir / "KPOINTS", "w") as f:
        f.write(f"Automatic\n0\nGamma\n{kpts[0]} {kpts[1]} {kpts[2]}\n0 0 0\n")

    with open(outdir / "POTCAR_ORDER", "w") as f:
        f.write(" ".join(species) + "\n")
