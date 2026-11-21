import numpy as np

from openeo_mountains_snow.config import SCA_RECONSTRUCTION_UDF

CLOUD, N_RANGES = 205, 10


def sca_cube(make_cube, snow, cp=0, occ=0):
    """Bands as built by snow_cover_area_cube: snow, scf, N_RANGES cp maps, N_RANGES occurrence maps."""
    bands = {"snow": snow, "scf": np.full_like(snow, 50)}
    bands |= {f"cp{i}": np.full_like(snow, cp) for i in range(N_RANGES)}
    bands |= {f"occ{i}": np.full_like(snow, occ) for i in range(N_RANGES)}
    return make_cube(bands)


def history(n_days=7):
    pattern = np.where(np.indices((6, 6)).sum(axis=0) % 3 == 0, 100.0, 0.0)
    return np.repeat(pattern[None], n_days, axis=0)


def test_cloudy_pixel_filled_from_similar_history(make_cube, run_udf):
    snow = history(13)  # needs more than 10 similar scenes
    snow[0] = 100  # fully snow-covered day: dissimilar, must not block the similar scenes
    snow[-1, 2, 3] = CLOUD
    cube = sca_cube(make_cube, snow)
    result = run_udf(SCA_RECONSTRUCTION_UDF, cube, {"n_days_to_reconstruct": 1})
    assert result.dims == ("t", "bands", "y", "x")
    assert list(result.bands.values) == ["reconstructed_snow"]
    assert list(result.t.values) == [cube.t.values[-1]]
    np.testing.assert_array_equal(result.values[0, 0], history()[-1])


def test_cloudy_pixel_filled_from_scf_probability(make_cube, run_udf):
    snow = history()
    snow[:-1] = 100  # no similar history -> HR step skipped, SCF/probability step must fill
    snow[-1, 2, 3] = CLOUD
    result = run_udf(SCA_RECONSTRUCTION_UDF, sca_cube(make_cube, snow, cp=1.0, occ=20), {"n_days_to_reconstruct": 1})
    assert result.values[0, 0, 2, 3] == 100


def test_nan_on_one_day_is_cloud_nan_on_all_days_is_no_data(make_cube, run_udf):
    snow = history(13)
    snow[-1, 2, 3] = np.nan  # missing on one day -> cloud -> filled from history
    snow[:, 0, 0] = np.nan  # missing on every day -> no data
    result = run_udf(SCA_RECONSTRUCTION_UDF, sca_cube(make_cube, snow), {"n_days_to_reconstruct": 1})
    assert result.values[0, 0, 2, 3] == history()[-1, 2, 3]
    assert result.values[0, 0, 0, 0] == 255


def test_default_reconstructs_every_day(make_cube, run_udf):
    result = run_udf(SCA_RECONSTRUCTION_UDF, sca_cube(make_cube, history()), {})
    assert result.sizes["t"] == history().shape[0]


def test_mostly_cloudy_modis_scene_is_dropped():
    from openeo_mountains_snow.snow_cover_area.udfs.snow_cover_area_reconstruction_udf import modis_blocks
    labels = np.arange(21)  # one HR pixel per LR pixel
    modis = np.full(21, CLOUD, dtype=float)
    modis[0] = 40  # 20 of 21 LR pixels cloudy: 95.2 % > 95 %
    assert modis_blocks(modis, labels, 21)[0] == 40
    assert (modis_blocks(modis, labels, 21, cloud_thres=95) == CLOUD).all()


def test_scf_interpolation_uses_dates_not_positions():
    from openeo_mountains_snow.snow_cover_area.udfs.snow_cover_area_reconstruction_udf import interpolate_scf
    scf = np.array([[0.0], [CLOUD], [90.0]])  # dates: day 0, day 1, day 9
    out = interpolate_scf(scf, np.zeros_like(scf), np.full_like(scf, 100), [0, 1, 9])
    assert out[1, 0] == 10


def test_clear_days_unchanged(make_cube, run_udf):
    cube = sca_cube(make_cube, history())
    result = run_udf(SCA_RECONSTRUCTION_UDF, cube, {"n_days_to_reconstruct": 3})
    np.testing.assert_array_equal(result.values[:, 0], history()[-3:])


def test_single_timestep_keeps_single_map(make_cube, run_udf):
    snow = history(1)
    result = run_udf(SCA_RECONSTRUCTION_UDF, sca_cube(make_cube, snow), {})
    assert result.sizes["t"] == 1
    assert list(result.bands.values) == ["reconstructed_snow"]
    np.testing.assert_array_equal(result.values[:, 0], snow)
