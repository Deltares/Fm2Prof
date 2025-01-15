import numpy as np
import pytest


from fm2prof.cross_section import CrossSection
import pickle

from tests.TestUtils import TestUtils

css_test_dir = "cross_sections"

test_cases = [
    dict(
        name="waal_1_40147.826",
        css_z=np.array(
            [
                -0.9141,
                -0.2808732,
                0.3523536,
                0.9855804,
                1.61880719,
                2.25203399,
                2.88526079,
                3.51848759,
                4.15171439,
                4.78494119,
                4.78494119,
                4.83562453,
                5.0825522,
                5.35312778,
                5.639405,
                5.94391206,
                6.23998542,
                6.54804721,
                6.84868257,
                7.13847582,
                7.38282862,
                7.63388787,
                7.89941093,
                8.18003839,
                8.4205877,
                8.68096833,
                8.90767707,
                9.14537943,
                9.39478408,
                9.62215027,
                9.81724704,
                10.01426908,
                10.2297726,
                10.46081785,
                10.71071257,
                10.99499479,
                11.29960472,
                11.62194869,
                11.95376401,
                12.20920782,
                12.26486054,
                12.27733635,
            ]
        ),
        css_total_volume=np.array(
            [
                0.0,
                202411.92430511,
                440863.87373976,
                714537.17597224,
                1023088.46888726,
                1370281.34820867,
                1746175.98609244,
                2142631.45969038,
                2559469.10395432,
                2991123.22671975,
                2991123.22671975,
                3028190.8376371,
                3218864.47242984,
                3430109.5158954,
                3655876.28183273,
                3898019.94718616,
                4136564.29328984,
                4391422.83609613,
                4652111.1476516,
                4914018.51771363,
                5139429.93097691,
                5372719.6548311,
                5620726.47479325,
                5884181.24648082,
                6110863.38850093,
                6359954.26342078,
                6580688.49745023,
                6813452.44399827,
                7062291.71909494,
                7293350.75974847,
                7491616.26087858,
                7691838.29814542,
                7910841.99672851,
                8145639.84965338,
                8399593.31489642,
                8693502.87558943,
                9016339.67243312,
                9360662.51019234,
                9715102.4974322,
                9987963.61958959,
                10047410.99351798,
                10060737.46221183,
            ]
        ),
        crest_level=4.573300167187546,
        extra_total_volume=689298.2236775636,
    )
]


class Test_generate_cross_section_instance:
    def test_when_wrong_input_dict_is_given_then_expected_exception_risen(self):
        # 1. Set up test data
        test_css_name = "dummy_css"
        css_data = {"id": test_css_name}

        # 2. Set expectations
        expected_error = "'Input data does not have all required keys'"

        # 3. Run test
        with pytest.raises(KeyError) as e_info:
            CrossSection(data=css_data)

        # 4. Verify final expectations
        error_message = str(e_info.value)
        assert error_message == expected_error, (
            "" + "Expected exception message {},".format(expected_error) + " retrieved {}".format(error_message)
        )

    def test_when_correct_input_dict_is_given_CrossSection_initialises(self):
        # 1. Set up test data
        tdir = TestUtils.get_local_test_data_dir(css_test_dir)
        with open(tdir.joinpath(f"{test_cases[0].get('name')}.pickle"), "rb") as f:
            css_data = pickle.load(f)

        # 2. Set expectations
        # 3. Run test
        css = CrossSection(data=css_data)

        # 4. Verify final expectations
        assert css.length == css_data.get("length")


class Test_cross_section_construction:
    def test_build_geometry(self):
        # 1. Set up test data
        test_case: dict = test_cases[0]
        tdir = TestUtils.get_local_test_data_dir(css_test_dir)
        with open(tdir.joinpath(f"{test_case.get('name')}.pickle"), "rb") as f:
            css_data = pickle.load(f)

        tol = 1e-6

        # 2. Set expectations for
        css_z: np.array = test_case.get("css_z")
        css_total_volume: np.array = test_case.get("css_total_volume")

        # 3. Run test
        css = CrossSection(data=css_data)
        css.build_geometry()

        # 4. Verify final expectations
        assert len(css_z) == len(css._css_z)

        assert all([abs(css_z[i] - css._css_z[i]) < tol for i in range(len(css_z))])
        assert all([abs(css_total_volume[i] - css._css_total_volume[i]) < tol for i in range(len(css_total_volume))])

    def test_calculate_correction(self):
        # 1. Set up test data
        test_case: dict = test_cases[0]
        tdir = TestUtils.get_local_test_data_dir(css_test_dir)
        with open(tdir.joinpath(f"{test_case.get('name')}.pickle"), "rb") as f:
            css_data = pickle.load(f)

        tol = 1e-6

        # 2. Set expectations for
        crest_level: float = test_case.get("crest_level")  # type: ignore
        extra_total_volume: np.ndarray = test_case.get("extra_total_volume")  # type: ignore

        # 3. Run test
        css = CrossSection(data=css_data)
        css.build_geometry()
        css.calculate_correction()

        # 4. Verify final expectations
        assert abs(crest_level - css.crest_level) < tol
        assert abs(extra_total_volume - css.extra_total_volume) < tol

    def test_reduce_points(self):
        # 1. Set up test data
        test_case: dict = test_cases[0]
        tdir = TestUtils.get_local_test_data_dir(css_test_dir)
        with open(tdir.joinpath(f"{test_case.get('name')}.pickle"), "rb") as f:
            css_data = pickle.load(f)

        # 2. Run test
        css = CrossSection(data=css_data)
        css.build_geometry()
        css.calculate_correction()
        css.reduce_points(count_after=20)

        # 3. Verify final expectations
        assert len(css.z) == 20
