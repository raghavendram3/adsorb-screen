from __future__ import annotations

from pathlib import Path
import numpy as np
from ase import Atoms
from ase.build import molecule as ase_molecule
from ase.collections import g2
from ase.data import chemical_symbols, covalent_radii
from ase.io import read


def _ch2i2() -> Atoms:
    r_ci, r_ch = 2.14, 1.09
    a, b = np.radians(113.0) / 2, np.radians(112.0) / 2
    return Atoms(
        "CI2H2",
        positions=[
            (0, 0, 0),
            (r_ci * np.sin(a), 0, r_ci * np.cos(a)),
            (-r_ci * np.sin(a), 0, r_ci * np.cos(a)),
            (0, r_ch * np.sin(b), -r_ch * np.cos(b)),
            (0, -r_ch * np.sin(b), -r_ch * np.cos(b)),
        ],
    )


def _ch2i() -> Atoms:
    r_ci, r_ch, t = 2.05, 1.08, np.radians(120.0)
    return Atoms(
        "CIH2",
        positions=[
            (0, 0, 0),
            (r_ci, 0, 0),
            (r_ch * np.cos(t), r_ch * np.sin(t), 0),
            (r_ch * np.cos(t), -r_ch * np.sin(t), 0),
        ],
    )


def _ch2() -> Atoms:
    r, h = 1.08, np.radians(134.0) / 2
    return Atoms(
        "CH2",
        positions=[
            (0, 0, 0),
            (r * np.sin(h), 0, r * np.cos(h)),
            (-r * np.sin(h), 0, r * np.cos(h)),
        ],
    )


def _ch2oo() -> Atoms:
    r_co, r_oo, r_ch = 1.27, 1.35, 1.08
    o1 = np.array([r_co, 0, 0])
    o2 = o1 + r_oo * np.array([np.cos(np.radians(62.0)), np.sin(np.radians(62.0)), 0])
    t = np.radians(120.0)
    return Atoms(
        "COOH2",
        positions=[
            (0, 0, 0),
            o1,
            o2,
            (r_ch * np.cos(t), r_ch * np.sin(t), 0),
            (r_ch * np.cos(t), -r_ch * np.sin(t), 0),
        ],
    )


BUILTIN_MOLECULES: dict[str, tuple[callable, int]] = {
    "CH2I2": (_ch2i2, 0),
    "CH2I": (_ch2i, 1),
    "CH2": (_ch2, 2),
    "CH2OO": (_ch2oo, 0),
}

# Standard ground-state unpaired electrons for isolated atoms
ATOM_UNPAIRED: dict[str, int] = {
    "H": 1, "Li": 1, "B": 1, "C": 2, "N": 3, "O": 2, "F": 1,
    "Na": 1, "Al": 1, "Si": 2, "P": 3, "S": 2, "Cl": 1,
    "K": 1, "Br": 1, "I": 1,
}


def _from_smiles(smiles: str) -> Atoms:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise SystemExit(
            "Install rdkit (`pip install adsorb-screen[smiles]`) to build molecules from SMILES."
        ) from exc

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if mol is None:
        raise SystemExit(f"Invalid SMILES string: '{smiles}'")
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    AllChem.UFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    positions = conf.GetPositions()
    return Atoms(symbols=symbols, positions=positions)


def load_molecule(spec: str, unpaired_override: int | None = None) -> tuple[str, Atoms, int]:
    """Load a molecule/atom from built-ins, file path, SMILES, single element, or ASE G2."""
    path = Path(spec)
    if path.is_file():
        mol = read(str(path))
        mol.set_cell(None)
        mol.pbc = False
        n_unp = unpaired_override if unpaired_override is not None else 0
        return path.stem, mol, n_unp

    if spec in BUILTIN_MOLECULES:
        builder, default_unp = BUILTIN_MOLECULES[spec]
        n_unp = unpaired_override if unpaired_override is not None else default_unp
        return spec, builder(), n_unp

    if spec.startswith("smiles:"):
        smiles_str = spec.split("smiles:", 1)[1]
        mol = _from_smiles(smiles_str)
        safe_name = "".join(c if c.isalnum() else "_" for c in smiles_str).strip("_")
        n_unp = unpaired_override if unpaired_override is not None else 0
        return f"smiles_{safe_name}", mol, n_unp

    if spec in chemical_symbols and spec != "X":
        mol = Atoms(spec, positions=[(0.0, 0.0, 0.0)])
        default_unp = ATOM_UNPAIRED.get(spec, 0)
        n_unp = unpaired_override if unpaired_override is not None else default_unp
        return spec, mol, n_unp

    if spec in g2.names:
        mol = ase_molecule(spec)
        mol.set_cell(None)
        mol.pbc = False
        inferred_unp = int(round(abs(mol.get_initial_magnetic_moments().sum())))
        n_unp = unpaired_override if unpaired_override is not None else inferred_unp
        return spec, mol, n_unp

    raise SystemExit(
        f"Unknown molecule '{spec}'. Supported inputs:\n"
        f"  1. Structure file (.xyz, POSCAR, .sdf, .cif)\n"
        f"  2. Built-in presets: {', '.join(BUILTIN_MOLECULES)}\n"
        f"  3. Any chemical element symbol (e.g., H, O, N, I, Pt)\n"
        f"  4. Any ASE G2 molecule name (e.g., CO, CO2, NH3, H2O, CH4, CH3OH, C2H4)\n"
        f"  5. SMILES string prefixed with 'smiles:' (e.g., 'smiles:CCO')"
    )


def build_dimer(symbol: str, box: float = 20.0) -> Atoms:
    """Build a homonuclear diatomic molecule X2 for half-dimer gas reference."""
    z = Atoms(symbol).numbers[0]
    r0 = 2.67 if symbol == "I" else float(2.0 * covalent_radii[z])
    dimer = Atoms(
        f"{symbol}2",
        positions=[(0.0, 0.0, 0.0), (0.0, 0.0, r0)],
        cell=[box, box, box],
        pbc=True,
    )
    dimer.center()
    return dimer
