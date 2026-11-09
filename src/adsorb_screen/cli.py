from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import warnings

import numpy as np
import pandas as pd
from ase.io import write
from fairchem.data.oc.core import Adsorbate, Slab
from fairchem.data.oc.core.multi_adsorbate_slab_config import MultipleAdsorbateSlabConfig

from .modes import binding_modes, describe_modes, orient_for_binding
from .molecules import BUILTIN_MOLECULES, build_dimer, load_molecule
from .relax import (
    OPTIMIZERS,
    find_anomalies,
    get_calculator,
    make_relax_job,
    min_ads_slab_distance,
    relax_gas_molecule,
)
from .slab import BUILTIN_BULKS, load_bulk_atoms, slab_from_bulk, slab_from_file
from .vasp import INCAR_GAS, INCAR_SLAB, write_vasp_dir

warnings.filterwarnings("ignore", category=UserWarning)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="adsorb-screen",
        description="General-purpose AdsorbML screening of molecules, radicals, and atoms on any catalytic surface.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g_mol = p.add_argument_group("Adsorbates (Molecules / Radicals / Atoms)")
    g_mol.add_argument(
        "--molecules",
        nargs="+",
        default=["CH2I2"],
        help="Built-ins, element symbols (O, H, I), ASE G2 names (CO, NH3, CH3OH), 'smiles:...', or structure files.",
    )
    g_mol.add_argument(
        "--unpaired",
        type=int,
        default=None,
        help="Override number of unpaired electrons for open-shell molecules.",
    )
    g_mol.add_argument(
        "--no-bidentate",
        dest="bidentate",
        action="store_false",
        help="Disable automatic bidentate pair binding modes.",
    )
    g_mol.add_argument(
        "--include-h-modes",
        action="store_true",
        help="Include H atoms when generating monodentate binding modes.",
    )
    g_mol.add_argument(
        "--modes",
        nargs="+",
        default=None,
        help="Filter specific binding mode names (e.g., I1 C0 I1-I2).",
    )
    g_mol.add_argument(
        "--atom-reference",
        default="half_dimer",
        choices=["half_dimer", "atom"],
        help="Gas-phase reference for single-atom adsorbates: 1/2 E(X2) or free atom.",
    )
    g_mol.add_argument(
        "--list-modes",
        action="store_true",
        help="Inspect generated binding modes and exit without running ML relaxations.",
    )

    g_slab = p.add_argument_group("Catalytic Surface / Slab")
    g_slab.add_argument(
        "--slab-file",
        default=None,
        help="Path to a pre-built surface slab (POSCAR, CIF, xyz).",
    )
    g_slab.add_argument(
        "--bulk",
        default="alpha-quartz",
        help=f"Bulk crystal when --slab-file is omitted: preset ({', '.join(BUILTIN_BULKS)}), "
             f"element/formula (Pt, Cu, Au, TiO2), MP-ID (mp-126), or bulk crystal file.",
    )
    g_slab.add_argument("--crystalstructure", default=None, help="ASE crystalstructure (fcc, bcc, hcp, rocksalt, etc.).")
    g_slab.add_argument("--lattice-a", type=float, default=None, help="Custom lattice constant 'a' (A) for ASE bulk.")
    g_slab.add_argument("--lattice-c", type=float, default=None, help="Custom lattice constant 'c' (A) for ASE bulk.")
    g_slab.add_argument("--millers", type=int, nargs=3, default=[0, 0, 1], help="Miller indices (h k l) for bulk cleaving.")
    g_slab.add_argument("--termination", type=int, default=None, help="Slab termination index (default: lowest broken bonds).")
    g_slab.add_argument("--min-ab", type=float, default=8.0, help="Minimum lateral slab dimension (A) when cleaving bulk.")
    g_slab.add_argument("--surface-depth", type=float, default=2.5, help="Top depth (A) tagged as surface for --slab-file.")
    g_slab.add_argument("--fixed-depth", type=float, default=6.0, help="Bottom depth (A) frozen as bulk (0 = no fixed atoms).")
    g_slab.add_argument("--metallic", action="store_true", help="Write metallic smearing (ISMEAR=1, SIGMA=0.2) in VASP INCARs.")

    g_ads = p.add_argument_group("AdsorbML Sampling & Relaxation")
    g_ads.add_argument("--num-placements", type=int, default=50, help="Number of placements per binding mode.")
    g_ads.add_argument("--steps", type=int, default=300, help="Maximum optimizer steps per placement.")
    g_ads.add_argument("--fmax", type=float, default=0.02, help="Force convergence criterion (eV/A).")
    g_ads.add_argument("--optimizer", default="LBFGS", choices=list(OPTIMIZERS), help="ASE geometry optimizer.")
    g_ads.add_argument("--desorb-dist", type=float, default=4.5, help="Max min(adsorbate-slab) distance (A) before flagging desorption.")
    g_ads.add_argument(
        "--placement-mode",
        default="random_site_heuristic_placement",
        choices=["random_site_heuristic_placement", "heuristic", "random"],
    )
    g_ads.add_argument(
        "--place-on-unrelaxed-slab",
        action="store_true",
        help="Place adsorbates on the unrelaxed input slab instead of the ML-relaxed clean slab.",
    )
    g_ads.add_argument("--save-trajectories", action="store_true", help="Save .traj files for all relaxations.")
    g_ads.add_argument("--seed", type=int, default=0, help="Random seed for reproducible placements.")

    g_ml = p.add_argument_group("MLIP Model")
    g_ml.add_argument("--model", default="uma-s-1p2p1", help="FAIRChem / UMA checkpoint name.")
    g_ml.add_argument("--task", default="oc25", choices=["oc25", "oc20", "omat", "omol", "odac", "omc"])
    g_ml.add_argument("--device", default=None, choices=["cpu", "cuda"])

    g_out = p.add_argument_group("Output")
    g_out.add_argument("--outdir", default="adsorbml_results", help="Root output directory.")
    g_out.add_argument("--top-k", type=int, default=5, help="Number of top candidates to export to VASP/traj.")
    g_out.add_argument("--no-vasp", action="store_true", help="Skip generating VASP input directories.")

    return p.parse_args()


def run_molecule(
    name: str,
    mol,
    unpaired: int,
    args: argparse.Namespace,
    outdir: Path,
    calc,
    relax,
    place_slab: Slab,
    relaxed_slab,
    e_slab: float,
) -> dict | None:
    mol_dir = outdir / name
    mol_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n################ {name} ({mol.get_chemical_formula()}) ################")

    # 1. Relax in gas phase FIRST so binding modes use relaxed bond lengths/angles
    gas_boxed, e_gas = relax_gas_molecule(mol, calc)
    write(str(mol_dir / f"{name}_gas_relaxed.xyz"), gas_boxed)
    print(f"E({name}, gas) = {e_gas:.4f} eV")

    mol_relaxed = gas_boxed.copy()
    mol_relaxed.set_cell(None)
    mol_relaxed.pbc = False

    # 2. Compute symmetry-unique binding modes on the relaxed gas geometry
    modes = binding_modes(
        mol_relaxed, bidentate=args.bidentate, include_h=args.include_h_modes
    )
    if args.modes:
        modes = {k: v for k, v in modes.items() if k in args.modes}
    describe_modes(name, mol_relaxed, modes)
    if not modes:
        print(f"  No matching binding modes for {name}; skipping.")
        return None

    # 3. Determine gas-phase reference energy
    single_atom = len(mol_relaxed) == 1
    e_ref, ref_label, dimer = e_gas, f"{name}(g)", None
    if single_atom:
        sym = mol_relaxed[0].symbol
        dimer_init = build_dimer(sym)
        dimer, e_dimer = relax_gas_molecule(dimer_init, calc)
        write(str(mol_dir / f"{sym}2_gas_relaxed.xyz"), dimer)
        print(
            f"E({sym}2, gas) = {e_dimer:.4f} eV (bond {dimer.get_distance(0, 1):.3f} A); "
            f"1/2 E({sym}2) - E({sym}) = {e_dimer / 2 - e_gas:.3f} eV"
        )
        if args.atom_reference == "half_dimer":
            e_ref, ref_label = e_dimer / 2.0, f"1/2 {sym}2(g)"
    elif unpaired > 0:
        print(
            f"  NOTE: {name} is open-shell ({unpaired} unpaired e-); non-spin-polarized MLIPs "
            f"carry extra reference uncertainty."
        )
    print(f"Reference for E_ads: {ref_label}")

    def extra_col(r: dict) -> dict:
        return {"E_ads_vs_free_atom_eV": r["E_ads_vs_gas"]} if single_atom and e_ref != e_gas else {}

    candidates, rows = [], []
    for mname, idx in modes.items():
        oriented, how = orient_for_binding(mol_relaxed, idx)
        print(f"\n=== {name} mode {mname} [{how}]: {args.num_placements} placements ===")
        ads = Adsorbate(adsorbate_atoms=oriented, adsorbate_binding_indices=idx)
        configs = MultipleAdsorbateSlabConfig(
            place_slab,
            [ads],
            num_configurations=args.num_placements,
            mode=args.placement_mode,
        ).atoms_list

        n_valid, kinds = 0, {}
        for k, init in enumerate(configs):
            res = relax(init, f"{name} {mname} #{k + 1}")
            anomalies = find_anomalies(init, res["atoms"], relaxed_slab, args.desorb_dist)
            res.update(
                mode=mname,
                E_ads=res["energy"] - e_slab - e_ref,
                E_ads_vs_gas=res["energy"] - e_slab - e_gas,
                anomalies=anomalies,
                min_dist=min_ads_slab_distance(res["atoms"]),
            )
            rows.append(
                {
                    "mode": mname,
                    "E_ads_eV": res["E_ads"],
                    **extra_col(res),
                    "E_adslab_eV": res["energy"],
                    "converged": res["converged"],
                    "min_ads_slab_dist_A": res["min_dist"],
                    "anomalies": ",".join(anomalies) or "none",
                }
            )
            for a in anomalies:
                kinds[a] = kinds.get(a, 0) + 1
            if not anomalies:
                candidates.append(res)
                n_valid += 1
        print(f"  {n_valid} valid, {len(configs) - n_valid} rejected {kinds if kinds else ''}")

    pd.DataFrame(rows).sort_values("E_ads_eV").to_csv(mol_dir / "all_placements.csv", index=False)
    if not candidates:
        print(f"  No valid configurations for {name}; see {mol_dir / 'all_placements.csv'}")
        return None

    candidates.sort(key=lambda c: c["E_ads"])
    df = pd.DataFrame(
        [
            {
                "mode": c["mode"],
                "E_ads_eV": c["E_ads"],
                **extra_col(c),
                "E_adslab_eV": c["energy"],
                "converged": c["converged"],
                "min_ads_slab_dist_A": c["min_dist"],
            }
            for c in candidates
        ]
    )
    df.index.name = "rank"
    df.to_csv(mol_dir / "adsorption_energies.csv")

    top = candidates[: args.top_k]
    write(str(mol_dir / "top_candidates.traj"), [c["atoms"] for c in top])
    write(str(mol_dir / "all_valid_candidates.traj"), [c["atoms"] for c in candidates])
    for k, c in enumerate(top):
        write(str(mol_dir / f"rank{k}_{c['mode']}.vasp"), c["atoms"], format="vasp", direct=True)

    print(f"\n{name}: top configurations (E_ads = E_adslab - E_slab - E[{ref_label}]; negative = exothermic)")
    print(df.head(args.top_k).to_string(float_format=lambda x: f"{x:.3f}"))

    if not args.no_vasp:
        write_vasp_dir(
            gas_boxed,
            mol_dir / "vasp" / "gas",
            INCAR_GAS,
            kpts=(1, 1, 1),
            unpaired=unpaired,
            fix_moment=True,
        )
        if dimer is not None:
            write_vasp_dir(dimer, mol_dir / "vasp" / "gas_dimer", INCAR_GAS, kpts=(1, 1, 1))
        for k, c in enumerate(top):
            write_vasp_dir(
                c["atoms"],
                mol_dir / "vasp" / f"rank{k}_{c['mode']}",
                INCAR_SLAB,
                unpaired=unpaired,
                metallic_slab=args.metallic,
            )

    best_per_mode = df.groupby("mode")["E_ads_eV"].min().to_dict()
    summary = {
        "molecule": name,
        "formula": mol_relaxed.get_chemical_formula(),
        "unpaired_electrons": unpaired,
        "E_gas_eV": e_gas,
        "E_reference_eV": e_ref,
        "reference": ref_label,
        "E_slab_eV": e_slab,
        "placements": len(rows),
        "valid": len(candidates),
        "best_mode": top[0]["mode"],
        "best_E_ads_eV": top[0]["E_ads"],
        "best_per_mode_eV": best_per_mode,
    }
    with open(mol_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    args = parse_args()
    mols = [load_molecule(s, args.unpaired) for s in args.molecules]

    if args.list_modes:
        for name, mol, _ in mols:
            modes = binding_modes(mol, bidentate=args.bidentate, include_h=args.include_h_modes)
            if args.modes:
                modes = {k: v for k, v in modes.items() if k in args.modes}
            describe_modes(name, mol, modes)
        return

    random.seed(args.seed)
    np.random.seed(args.seed)

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    traj_dir = (outdir / "trajectories") if args.save_trajectories else None
    if traj_dir is not None:
        traj_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {args.model} | Task: {args.task} | Device: {args.device or 'auto'}")

    if args.slab_file:
        slab = slab_from_file(str(Path(args.slab_file).resolve()), args.surface_depth, args.fixed_depth)
    else:
        bulk_atoms = load_bulk_atoms(
            args.bulk,
            crystalstructure=args.crystalstructure,
            a=args.lattice_a,
            c=args.lattice_c,
        )
        slab = slab_from_bulk(
            bulk_atoms,
            tuple(args.millers),
            termination=args.termination,
            fixed_depth=args.fixed_depth,
            min_ab=args.min_ab,
        )

    write(str(outdir / "slab_input.vasp"), slab.atoms, format="vasp", direct=True)

    calc = get_calculator(args.model, args.task, args.device)
    relax = make_relax_job(calc, args.fmax, args.steps, args.optimizer, traj_dir)

    print("\nRelaxing clean slab:")
    slab_res = relax(slab.atoms, "clean slab")
    e_slab, relaxed_slab = slab_res["energy"], slab_res["atoms"]
    write(str(outdir / "slab_relaxed.vasp"), relaxed_slab, format="vasp", direct=True)

    if not args.no_vasp:
        write_vasp_dir(relaxed_slab, outdir / "vasp_slab", INCAR_SLAB, metallic_slab=args.metallic)

    # Default: place on the relaxed slab to avoid artificial surface_changed anomalies
    place_slab = slab if args.place_on_unrelaxed_slab else Slab(slab_atoms=relaxed_slab.copy())

    summaries = []
    for name, mol, unpaired in mols:
        s = run_molecule(
            name, mol, unpaired, args, outdir, calc, relax, place_slab, relaxed_slab, e_slab
        )
        if s is not None:
            summaries.append(s)

    if summaries:
        tab = pd.DataFrame(
            [
                {
                    "molecule": s["molecule"],
                    "best_mode": s["best_mode"],
                    "E_ads_eV": s["best_E_ads_eV"],
                    "reference": s["reference"],
                    "valid": s["valid"],
                    "placements": s["placements"],
                    "unpaired_e": s["unpaired_electrons"],
                }
                for s in summaries
            ]
        )
        tab.to_csv(outdir / "summary_all_molecules.csv", index=False)
        print("\n================ Summary ================")
        print(tab.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\nAll output written to: {outdir}")


if __name__ == "__main__":
    main()
