from __future__ import annotations

from pathlib import Path
import time
import numpy as np
from ase import Atoms
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
