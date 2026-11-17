from __future__ import annotations

from pathlib import Path
import time
import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.optimize import BFGS, FIRE, LBFGS
from fairchem.data.oc.utils import DetectTrajAnomaly


OPTIMIZERS = {"LBFGS": LBFGS, "BFGS": BFGS, "FIRE": FIRE}


def get_calculator(model: str, task: str, device: str | None = None):
    from fairchem.core import FAIRChemCalculator, pretrained_mlip
    predict_unit = pretrained_mlip.get_predict_unit(model, device=device)
    return FAIRChemCalculator(predict_unit, task_name=task)


def make_relax_job(
    calc,
    fmax: float = 0.02,
    steps: int = 300,
    optimizer_name: str = "LBFGS",
    traj_dir: Path | None = None,
):
    opt_cls = OPTIMIZERS.get(optimizer_name.upper(), LBFGS)

    def relax(initial_atoms: Atoms, label: str) -> dict:
        atoms = initial_atoms.copy()
        atoms.calc = calc
        traj_path = None
        if traj_dir is not None:
            traj_dir.mkdir(parents=True, exist_ok=True)
            safe_label = label.replace(" #", "_").replace(" ", "_")
            traj_path = str(traj_dir / f"{safe_label}.traj")
        t0 = time.time()
        opt = opt_cls(atoms, logfile=None, trajectory=traj_path)
        converged = bool(opt.run(fmax=fmax, steps=steps))
        energy = float(atoms.get_potential_energy())
        atoms.calc = None
        print(
            f"  {label:>26s}: E = {energy:12.4f} eV, {opt.nsteps:4d} steps, "
            f"{'converged' if converged else 'NOT converged'}, {time.time() - t0:6.1f} s",
            flush=True,
        )
        return {
            "initial": initial_atoms,
            "atoms": atoms,
            "energy": energy,
            "converged": converged,
            "trajectory": traj_path,
        }

    return relax


def relax_gas_molecule(
    mol: Atoms, calc, box: float = 20.0, fmax: float = 0.01, steps: int = 500
) -> tuple[Atoms, float]:
    gas = mol.copy()
    gas.set_cell([box, box, box])
    gas.center()
    gas.pbc = True
    gas.calc = calc
    if len(gas) > 1:
        LBFGS(gas, logfile=None).run(fmax=fmax, steps=steps)
    e = float(gas.get_potential_energy())
    gas.calc = None
    # Strip cell for clean orientation processing while keeping relaxed coordinates
    gas_no_cell = gas.copy()
    gas_no_cell.set_cell(None)
    gas_no_cell.pbc = False
    return gas, e


def min_ads_slab_distance(atoms: Atoms) -> float:
    tags = atoms.get_tags()
    ads = np.where(tags == 2)[0]
    sub = np.where(tags != 2)[0]
    if len(ads) == 0 or len(sub) == 0:
        return 0.0
    return float(min(atoms.get_distances(i, sub, mic=True).min() for i in ads))


def adsorption_geometry(
    atoms: Atoms, relaxed_slab: Atoms, initial: Atoms | None = None
) -> dict[str, float | int]:
    """Return geometric descriptors useful for interpreting an adsorption structure."""
    tags = atoms.get_tags()
    ads = np.where(tags == 2)[0]
    sub = np.where(tags != 2)[0]
    if len(ads) == 0 or len(sub) == 0:
        return {
            "min_ads_slab_distance_A": 0.0,
            "mean_ads_slab_distance_A": 0.0,
            "surface_contacts": 0,
            "adsorbate_height_above_slab_A": 0.0,
            "surface_rmsd_A": 0.0,
            "adsorbate_distortion_rmsd_A": 0.0,
        }

    distances = np.concatenate(
        [atoms.get_distances(i, sub, mic=True) for i in ads]
    )
    contacts = 0
    for i in ads:
        for j in sub:
            cutoff = 1.25 * (
                covalent_radii[atoms[i].number] + covalent_radii[atoms[j].number]
            )
            if atoms.get_distance(i, j, mic=True) <= cutoff:
                contacts += 1

    slab_top = float(np.max(relaxed_slab.positions[:, 2]))
    ads_com = atoms.positions[ads].mean(axis=0)
    surface_rmsd = 0.0
    if len(relaxed_slab) == len(sub):
        displacement = atoms.positions[sub] - relaxed_slab.positions
        surface_rmsd = float(np.sqrt(np.mean(np.sum(displacement**2, axis=1))))

    distortion = 0.0
    if initial is not None and len(ads) == int((initial.get_tags() == 2).sum()):
        initial_ads = np.where(initial.get_tags() == 2)[0]
        final_centered = atoms.positions[ads] - atoms.positions[ads].mean(axis=0)
        initial_centered = initial.positions[initial_ads] - initial.positions[initial_ads].mean(axis=0)
        distortion = float(
            np.sqrt(np.mean(np.sum((final_centered - initial_centered) ** 2, axis=1)))
        )

    return {
        "min_ads_slab_distance_A": float(distances.min()),
        "mean_ads_slab_distance_A": float(distances.mean()),
        "surface_contacts": contacts,
        "adsorbate_height_above_slab_A": float(ads_com[2] - slab_top),
        "surface_rmsd_A": surface_rmsd,
        "adsorbate_distortion_rmsd_A": distortion,
    }


def explain_top_configuration(
    top: dict, candidates: list[dict], relaxed_slab: Atoms
) -> dict:
    """Compare the best candidate's geometry with valid candidates.

    The returned reasons are geometric evidence associated with the energy ranking,
    not a replacement for an electronic-structure analysis.
    """
    top_geometry = adsorption_geometry(top["atoms"], relaxed_slab, top.get("initial"))
    geometries = [
        adsorption_geometry(c["atoms"], relaxed_slab, c.get("initial"))
        for c in candidates
    ]
    median = {
        key: float(np.median([g[key] for g in geometries]))
        for key in top_geometry
    }
    reasons = []
    if top["E_ads"] < float(np.median([c["E_ads"] for c in candidates])):
        reasons.append("more favorable adsorption energy than the valid-candidate median")
    if top_geometry["surface_contacts"] > median["surface_contacts"]:
        reasons.append("more adsorbate-surface contacts than the valid-candidate median")
    if top_geometry["surface_rmsd_A"] < median["surface_rmsd_A"]:
        reasons.append("less surface rearrangement than the valid-candidate median")
    if top_geometry["adsorbate_distortion_rmsd_A"] < median["adsorbate_distortion_rmsd_A"]:
        reasons.append("less adsorbate distortion than the valid-candidate median")
    if not reasons:
        reasons.append("its geometry is the lowest-energy valid structure; no single geometric descriptor dominates")

    return {
        "top_energy_eV": float(top["E_ads"]),
        "top_mode": top["mode"],
        "top_geometry": top_geometry,
        "valid_candidate_medians": median,
        "reasons": reasons,
    }


def find_anomalies(
    initial: Atoms, final: Atoms, relaxed_slab: Atoms, desorb_dist: float
) -> list[str]:
    det = DetectTrajAnomaly(
        initial, final, atoms_tag=initial.get_tags(), final_slab_atoms=relaxed_slab
    )
    found = []
    if (initial.get_tags() == 2).sum() > 1 and det.is_adsorbate_dissociated():
        found.append("dissociated")
    if min_ads_slab_distance(final) > desorb_dist:
        found.append("desorbed")
    if det.has_surface_changed():
        found.append("surface_changed")
    if det.is_adsorbate_intercalated():
        found.append("intercalated")
    return found
