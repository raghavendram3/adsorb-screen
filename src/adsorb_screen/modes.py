from __future__ import annotations

import itertools
import numpy as np
from ase import Atoms
from ase.neighborlist import NeighborList, natural_cutoffs


def safe_rotate(atoms: Atoms, v1: np.ndarray, v2: np.ndarray, center: np.ndarray) -> None:
    """Rotate vector v1 onto v2 around center, safely handling collinear/antiparallel vectors."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return
    u1, u2 = v1 / n1, v2 / n2
    if np.allclose(u1, u2, atol=1e-6):
        return
    if np.allclose(u1, -u2, atol=1e-6):
        ortho = np.array([1.0, 0.0, 0.0]) if abs(u1[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(u1, ortho)
        atoms.rotate(180.0, axis, center=center)
    else:
        atoms.rotate(u1, u2, center=center)


def bonded_neighbors(mol: Atoms, mult: float = 1.15) -> list[list[int]]:
    if len(mol) <= 1:
        return [[]]
    nl = NeighborList(natural_cutoffs(mol, mult=mult), self_interaction=False, bothways=True)
    nl.update(mol)
    return [list(nl.get_neighbors(i)[0]) for i in range(len(mol))]


def atom_fingerprint(mol: Atoms, i: int, tol: float = 0.08) -> tuple:
    """Distance-spectrum + coordination fingerprint for symmetry equivalence."""
    d = mol.get_distances(i, range(len(mol)))
    spectrum = tuple(
        sorted((mol[j].symbol, round(float(d[j]) / tol)) for j in range(len(mol)) if j != i)
    )
    return (mol[i].symbol,) + spectrum


def orient_for_binding(
    mol: Atoms, atoms: list[int], flat_threshold: float = 0.5
) -> tuple[Atoms, str]:
    """Orient molecule so binding atom(s) face -z (toward the catalytic surface)."""
    m = mol.copy()
    if len(m) == 1:
        return m, "atom"

    center = m.positions[atoms].mean(axis=0)
    if len(atoms) == 2:
        ax = m.positions[atoms[1]] - m.positions[atoms[0]]
        safe_rotate(m, ax, np.array([1.0, 0.0, 0.0]), center=center)
        others = [k for k in range(len(m)) if k not in atoms]
        if not others:
            return m, "pair"
        best, best_score = None, None
        for ang in np.arange(0.0, 360.0, 2.0):
            t = m.copy()
            t.rotate(ang, "x", center=center)
            dz = t.positions[others, 2] - t.positions[atoms, 2].mean()
            score = (round(float(dz.min()), 2), float(dz.mean()))
            if best_score is None or score > best_score:
                best, best_score = t, score
        return best, "pair"

    nbrs = bonded_neighbors(m)
    vec = np.zeros(3)
    for i in atoms:
        for j in nbrs[i]:
            u = m.positions[j] - m.positions[i]
            norm = np.linalg.norm(u)
            if norm > 1e-8:
                vec += u / norm

    if np.linalg.norm(vec) > flat_threshold:
        safe_rotate(m, -vec, np.array([0.0, 0.0, -1.0]), center=center)
        return m, "down"

    x = m.positions - m.positions.mean(axis=0)
    normal = np.linalg.eigh(x.T @ x)[1][:, 0]
    safe_rotate(m, normal, np.array([0.0, 0.0, 1.0]), center=center)
    return m, "flat"


def binding_modes(
    mol: Atoms,
    bidentate: bool = True,
    include_h: bool = False,
    pair_cutoff: float = 4.0,
) -> dict[str, list[int]]:
    """Find symmetry-unique monodentate and bidentate binding modes."""
    if len(mol) == 1:
        return {f"{mol[0].symbol}0": [0]}

    candidates = [
        i for i, a in enumerate(mol) if include_h or a.symbol != "H"
    ]
    if not candidates:
        candidates = list(range(len(mol)))

    modes: dict[str, list[int]] = {}
    seen: set[tuple] = set()
    for i in candidates:
        fp = atom_fingerprint(mol, i)
        if fp not in seen:
            seen.add(fp)
            modes[f"{mol[i].symbol}{i}"] = [i]

    if bidentate and len(candidates) >= 2:
        seen_pairs: set[tuple] = set()
        for i, j in itertools.combinations(candidates, 2):
            if mol.get_distance(i, j) > pair_cutoff:
                continue
            fp_pair = tuple(sorted([atom_fingerprint(mol, i), atom_fingerprint(mol, j)]))
            o, _ = orient_for_binding(mol, [i, j])
            others = [k for k in range(len(mol)) if k not in (i, j)]
            if others and (o.positions[others, 2] - o.positions[[i, j], 2].mean()).min() < 0.3:
                continue
            if fp_pair not in seen_pairs:
                seen_pairs.add(fp_pair)
                modes[f"{mol[i].symbol}{i}-{mol[j].symbol}{j}"] = [i, j]

    return modes


def describe_modes(name: str, mol: Atoms, modes: dict[str, list[int]]) -> None:
    print(f"\n{name} ({mol.get_chemical_formula()}): {len(modes)} binding mode(s)")
    for mname, idx in modes.items():
        o, how = orient_for_binding(mol, idx)
        z = o.positions[:, 2] - o.positions[idx, 2].mean()
        prof = ", ".join(
            f"{s}{k}:{dz:+.2f}" for k, (s, dz) in enumerate(zip(o.get_chemical_symbols(), z))
        )
        print(f"  {mname:<10s} [{how:4s}]  heights rel. to binding atom(s): {prof}")
