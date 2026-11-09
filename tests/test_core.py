import numpy as np
from adsorb_screen.modes import binding_modes, orient_for_binding, safe_rotate
from adsorb_screen.molecules import load_molecule
from adsorb_screen.slab import get_bulk_cn, load_bulk_atoms


def test_safe_rotate_antiparallel():
    _, mol, _ = load_molecule("CH2I2")
    v1 = np.array([0.0, 0.0, 1.0])
    v2 = np.array([0.0, 0.0, -1.0])
    safe_rotate(mol, v1, v2, center=np.zeros(3))
    assert not np.isnan(mol.positions).any()


def test_binding_modes_ch2i2():
    _, mol, _ = load_molecule("CH2I2")
    modes = binding_modes(mol, bidentate=True)
    # Expect C0 (flat), one unique I (down), and I1-I2 (bidentate pair)
    assert len(modes) == 3
    o_pair, how = orient_for_binding(mol, [1, 2])
    assert how == "pair"
    # C and H atoms must sit higher in z than the two binding I atoms
    z_i = o_pair.positions[[1, 2], 2].mean()
    assert (o_pair.positions[[0, 3, 4], 2] > z_i + 0.3).all()


def test_load_any_g2_and_atom():
    name_co, co, unp_co = load_molecule("CO")
    assert len(co) == 2 and unp_co == 0

    name_o, o_atom, unp_o = load_molecule("O")
    assert len(o_atom) == 1 and unp_o == 2


def test_bulk_cn_quartz_and_pt():
    quartz = load_bulk_atoms("alpha-quartz")
    cn_q = get_bulk_cn(quartz)
    assert cn_q["Si"] == 4 and cn_q["O"] == 2

    pt = load_bulk_atoms("Pt")
    cn_pt = get_bulk_cn(pt)
    assert cn_pt["Pt"] == 12
