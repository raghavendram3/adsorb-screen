from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.build import bulk as ase_bulk
from ase.constraints import FixAtoms
from ase.io import read
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.spacegroup import crystal
from fairchem.data.oc.core import Bulk, Slab


BUILTIN_BULKS: dict[str, callable] = {
    "alpha-quartz": lambda: crystal(
        ["Si", "O"],
        basis=[(0.4697, 0.0, 1 / 3), (0.4135, 0.2669, 0.2142)],
        spacegroup=152,
        cellpar=[4.9134, 4.9134, 5.4052, 90, 90, 120],
    ),
    "rutile-TiO2": lambda: crystal(
        ["Ti", "O"],
        basis=[(0.0, 0.0, 0.0), (0.3048, 0.3048, 0.0)],
        spacegroup=136,
        cellpar=[4.5937, 4.5937, 2.9587, 90, 90, 90],
    ),
    "MgO": lambda: ase_bulk("MgO", crystalstructure="rocksalt", a=4.212),
}


def load_bulk_atoms(
    bulk_spec: str,
    crystalstructure: str | None = None,
    a: float | None = None,
    c: float | None = None,
) -> Atoms:
    """Load bulk crystal from preset, file path, Materials Project ID, or ASE bulk."""
    if bulk_spec in BUILTIN_BULKS:
        return BUILTIN_BULKS[bulk_spec]()

    if bulk_spec.startswith("mp-"):
        try:
            from mp_api.client import MPRester
            from pymatgen.io.ase import AseAtomsAdaptor
        except ImportError as exc:
            raise SystemExit(
                "Install mp-api (`pip install adsorb-screen[mp]`) to fetch Materials Project IDs."
            ) from exc
        with MPRester() as mpr:
            struct = mpr.get_structure_by_material_id(bulk_spec, conventional_unit_cell=True)
        return AseAtomsAdaptor.get_atoms(struct)

    try:
        return read(bulk_spec)
    except Exception:
        pass

    try:
        kwargs = {}
        if crystalstructure:
            kwargs["crystalstructure"] = crystalstructure
        if a is not None:
            kwargs["a"] = a
        if c is not None:
            kwargs["c"] = c
        return ase_bulk(bulk_spec, **kwargs)
    except Exception as exc:
        raise SystemExit(
            f"Could not build bulk '{bulk_spec}'. Provide a valid structure file, "
            f"preset ({', '.join(BUILTIN_BULKS)}), ASE formula (e.g. Pt, Cu, Fe, Au, NaCl), "
            f"or Materials Project ID (mp-XXXX)."
        ) from exc


def get_bulk_cn(bulk_atoms: Atoms, mult: float = 1.15) -> dict[str, int]:
    """Dynamically determine ideal bulk coordination numbers per element."""
    rep = bulk_atoms.repeat((2, 2, 2))
    nl = NeighborList(natural_cutoffs(rep, mult=mult), self_interaction=False, bothways=True)
    nl.update(rep)
    cn_map: dict[str, int] = {}
    for idx, atom in enumerate(rep):
        cn = len(nl.get_neighbors(idx)[0])
        cn_map[atom.symbol] = max(cn_map.get(atom.symbol, 0), cn)
    return cn_map


def broken_bonds(atoms: Atoms, bulk_cn: dict[str, int], mult: float = 1.15) -> int:
    """Count under-coordinated bonds on surface atoms (tag == 1) for any crystal."""
    nl = NeighborList(natural_cutoffs(atoms, mult=mult), self_interaction=False, bothways=True)
    nl.update(atoms)
    surf = np.where(atoms.get_tags() == 1)[0]
    return int(
        sum(max(bulk_cn.get(atoms[k].symbol, 0) - len(nl.get_neighbors(k)[0]), 0) for k in surf)
    )


def apply_fixed_depth(atoms: Atoms, fixed_depth: float) -> Atoms:
    """Freeze atoms within fixed_depth (A) of the bottom of the slab."""
    z = atoms.positions[:, 2]
    thickness = float(z.max() - z.min())
    fixed = z < (z.min() + fixed_depth) if fixed_depth > 0 else np.zeros(len(atoms), dtype=bool)
    if fixed.all():
        raise SystemExit(
            f"--fixed-depth {fixed_depth} A fixes the entire slab (thickness {thickness:.2f} A)."
        )
    overlap = fixed & (atoms.get_tags() == 1)
    if overlap.any():
        print(f"  WARNING: {overlap.sum()} surface-tagged atoms are inside --fixed-depth.")
    atoms.set_constraint(FixAtoms(mask=fixed) if fixed.any() else [])
    free = ~fixed
    print(
        f"  Fixed {fixed.sum()} / {len(atoms)} atoms: bottom {fixed_depth:.2f} A of a "
        f"{thickness:.2f} A slab (relaxing {free.sum()} atoms with z >= {z[free].min():.2f} A)"
    )
    return atoms


def slab_from_bulk(
    bulk_atoms: Atoms,
    millers: tuple[int, int, int],
    termination: int | None = None,
    fixed_depth: float = 6.0,
    min_ab: float = 8.0,
) -> Slab:
    """Cleave bulk crystal along Miller indices, pick lowest broken-bond termination, and tile."""
    bulk_cn = get_bulk_cn(bulk_atoms)
    slabs = Slab.from_bulk_get_specific_millers(
        tuple(millers), bulk=Bulk(bulk_atoms=bulk_atoms), min_ab=min_ab
    )
    print(f"{len(slabs)} termination(s) for {bulk_atoms.get_chemical_formula()} {tuple(millers)}:")
    scores = []
    for k, s in enumerate(slabs):
        bb = broken_bonds(s.atoms, bulk_cn)
        scores.append(bb)
        print(f"  [{k}] {s}  under-coordinated surface bonds: {bb}")
    pick = termination if termination is not None else int(np.argmin(scores))
    print(f"Selected termination [{pick}]")
    slab = slabs[pick]
    if fixed_depth is not None:
        apply_fixed_depth(slab.atoms, fixed_depth)
    return slab


def slab_from_file(path: str, surface_depth: float, fixed_depth: float) -> Slab:
    """Load custom slab file and tag top surface_depth A as surface (tag=1)."""
    atoms = read(path)
    atoms.pbc = True
    z = atoms.positions[:, 2]
    surf = z > (z.max() - surface_depth)
    if surf.sum() == 0:
        raise SystemExit("Check --surface-depth: no atoms tagged as surface.")
    atoms.set_tags(surf.astype(int))
    vac = float(atoms.cell.lengths()[2] - (z.max() - z.min()))
    print(
        f"Slab {atoms.get_chemical_formula()} from {path}: {len(atoms)} atoms, "
        f"{surf.sum()} surface (top {surface_depth:.2f} A), ~{vac:.1f} A vacuum"
    )
    apply_fixed_depth(atoms, fixed_depth)
    if vac < 12.0:
        print("  WARNING: Vacuum is less than 12 A; periodic image interactions may occur.")
    return Slab(slab_atoms=atoms)
