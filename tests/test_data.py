import numpy as np

from pivbench.data import find_pairs, load_ground_truth, load_piv_challenge_case


def test_find_pairs_and_case(tmp_path):
    import tifffile

    a = (np.random.default_rng(0).random((16, 16)) * 255).astype(np.uint8)
    for name in ["A001a.tif", "A001b.tif", "I1_case.tif", "I2_case.tif"]:
        tifffile.imwrite(tmp_path / name, a)
    np.savez(tmp_path / "truth.npz", u=np.ones((16, 16)), v=np.zeros((16, 16)))
    pairs = {(p.name, q.name) for p, q in find_pairs(tmp_path)}
    assert ("A001a.tif", "A001b.tif") in pairs and ("I1_case.tif", "I2_case.tif") in pairs
    case = load_piv_challenge_case(tmp_path, truth_glob="truth.npz")
    s = case[0]
    assert s["img1"].dtype == np.float32 and s["img1"].max() <= 1.0
    assert s["u"].mean() == 1.0


def test_ground_truth_table(tmp_path):
    x, y = np.meshgrid(np.arange(0, 33, 4.0), np.arange(0, 33, 4.0))
    np.savetxt(tmp_path / "t.dat", np.column_stack([x.ravel(), y.ravel(), 0.1 * x.ravel(), 0 * x.ravel() + 1]),
               header='VARIABLES = "x" "y" "u" "v"')
    u, v = load_ground_truth(tmp_path / "t.dat", shape=(33, 33))
    np.testing.assert_allclose(u, 0.1 * np.arange(33)[None, :].repeat(33, 0), atol=1e-6)
