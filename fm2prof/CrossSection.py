"""
Contains CrossSection class
"""
import logging
import math
import os
from datetime import datetime, timedelta
from functools import reduce
from logging import Logger
from time import time
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.optimize as so
from scipy.integrate import cumtrapz

from fm2prof import Functions as FE
from fm2prof.common import FM2ProfBase, FrictionTable
from fm2prof.Import import FmModelData
from fm2prof.IniFile import IniFile
from fm2prof.MaskOutputFile import MaskOutputFile

from .lib import polysimplify as PS

pd.options.mode.chained_assignment = None  # default='warn'


class CrossSection(FM2ProfBase):
    """
    Use this class to derive cross-sections from fm_data (2D model results).
    See docs how to acquire fm_data and how to prepare a proper 2D model.
    """

    __cs_parameter_css_points = "MaximumPointsInProfile"
    __cs_parameter_transitionheight_sd = "SDTransitionHeight"
    __cs_parameter_velocity_threshold = "AbsoluteVelocityThreshold"
    __cs_parameter_relative_threshold = "RelativeVelocityThreshold"
    __cs_parameter_min_depth_storage = "MinimumDepthThreshold"
    __cs_parameter_plassen_timesteps = "LakeTimesteps"
    __cs_parameter_storagemethod_wli = "ExtrapolateStorage"
    __cs_parameter_bedlevelcriterium = "BedlevelCriterium"
    __cs_parameter_SDstorage = "SDCorrection"
    __cs_parameter_Frictionweighing = "FrictionweighingMethod"
    __cs_parameter_sdoptimisationmethod = "sdoptimisationmethod"
    __cs_parameter_skip_maps = "SkipMaps"
    __cs_parameter_floodplain_base_level = "SDFloodplainBase"
    __cs_parameter_minwidth = "MinimumTotalWidth"
    __logger = None

    def __init__(
        self,
        name: str,
        length: float,
        location: tuple,
        branchid="not defined",
        chainage=0,
        fm_data: FmModelData = None,
        logger: Logger = None,
        inifile: IniFile = None,
    ):
        """
        Arguments:
            InputParam_dict {Dictionary} -- [description]
            name {str} -- [description]
            length {float} -- [description]
            location {tuple} -- [description]

        Keyword Arguments:
            branchid {str} -- [description] (default: {"not defined"})
            chainage {int} -- [description] (default: {0})
        """
        super().__init__(logger=logger, inifile=inifile)

        # Cross-section meta data
        self.name = name  # cross-section id
        self.length = length  # 'vaklengte'
        self.location = location  # (x,y)
        self.branch = branchid  # name of 1D branch for cross-section
        self.chainage = chainage  # offset from beginning of branch
        self.__output_mask_list = []  # initialize output mask list.
        self._fm_data = fm_data  # dictionary with fmdata

        # Cross-section geometry
        self.z = []
        self.total_width = []
        self.flow_width = []
        self.section_widths = {"main": 0, "floodplain1": 0, "floodplain2": 0}
        self.friction_tables = dict()
        self.roughness_sections = np.array([])

        # delta h corrections ("summerdike option")
        self.crest_level = 0
        # in cross-section def. WAQ2PROF did crest - some fixed value.
        #  how to do here?
        self.floodplain_base = 0.0
        # note" 'to avoid numerical oscillation'. might need minimal value.
        # fixed or variable? Test!
        self.transition_height = 0.5
        self.extra_flow_area = 0.0
        self.extra_total_volume = 0.0
        self.extra_area_percentage = list()
        self.extra_total_area = 0
        self.extra_flow_area = 0

        # These attributes are used for non-reduced sets
        self._css_z = 0
        self._css_total_volume = 0
        self._css_total_volume_corrected = None
        self._css_flow_volume = 0
        self._css_flow_volume_corrected = None
        self._css_total_width = 0
        self._css_flow_width = 0
        self._css_volume_legacy = 0
        self._css_index_of_first_nonzero = 0
        self._fm_total_volume = 0
        self._fm_flow_volume = 0
        self._fm_wet_area = 0
        self._fm_flow_area = 0

        # flags
        self._css_is_corrected = False
        self._css_is_reduced = False

        # data structures
        self.__output_face_list = []
        self.__output_edge_list = []

        self._section_map = {
            "1": "main",
            "2": "floodplain1",
            "3": "floodplain2",
            "-999": "main",
        }

    @property
    def alluvial_width(self):
        for key in [1, "1", "main", "Main"]:
            try:
                return self.section_widths[key]
            except KeyError:
                pass
        return 0

    @property
    def nonalluvial_width(self):
        for key in [2, "2", "floodplain", "FloodPlain1"]:
            try:
                return self.section_widths[key]
            except KeyError:
                pass
        return 0

    @property
    def face_points_list(self):
        return self.__output_face_list

    @property
    def edge_points_list(self):
        return self.__output_edge_list

    def get_point_list(self, pointtype):
        if pointtype == "face":
            return self.face_points_list
        elif pointtype == "edge":
            return self.edge_points_list
        else:
            raise ValueError('pointtype must be "face" or "edge"')

    # Public functions
    def build_geometry(self):
        """
        Build 1D geometrical cross-section from FM data.

        :param fm_data: dict
        :return:
        """
        fm_data = self._fm_data

        # Unpack FM data
        waterlevel = fm_data["waterlevel"].iloc[
            :, self.get_parameter(self.__cs_parameter_skip_maps) :
        ]
        waterdepth = fm_data["waterdepth"].iloc[
            :, self.get_parameter(self.__cs_parameter_skip_maps) :
        ]
        velocity = fm_data["velocity"].iloc[
            :, self.get_parameter(self.__cs_parameter_skip_maps) :
        ]
        area = fm_data["area"]
        bedlevel = fm_data["bedlevel"]

        # Convert area to a matrix for matrix operations
        # (much more efficient than for-loops)
        area_matrix = pd.DataFrame(index=area.index)
        for t in waterdepth:
            area_matrix[t] = area

        bedlevel_matrix = pd.DataFrame(index=bedlevel.index)
        for t in waterdepth:
            bedlevel_matrix[t] = bedlevel

        # Retrieve the water-depth
        # & water level nearest to the cross-section location
        self.set_logger_message("Retrieving centre point values")
        (centre_depth, centre_level) = FE.get_centre_values(
            self.location, fm_data["x"], fm_data["y"], waterdepth, waterlevel
        )

        # apply rolling average over the velocities
        # to smooth out extreme values
        velocity = velocity.rolling(
            window=10, min_periods=1, center=True, axis=1
        ).mean()

        # Identify river lakes (plassen)
        self.set_logger_message("Identifying lakes")
        (
            plassen_mask,
            wet_not_plas_mask,
            plassen_depth_correction,
        ) = self._identify_lakes(
            waterdepth
        )  # plassen_mask needed for arrays in output

        # Masks for wet and flow cells (stroomvoeringscriteria)
        self.set_logger_message("Seperating flow from storage")
        flow_mask = self._distinguish_flow_from_storage(waterdepth, velocity)

        # Calculate area and volume as function of waterlevel & waterdepth
        self._fm_wet_area = np.nansum(area_matrix[wet_not_plas_mask], axis=0)
        self._fm_flow_area = np.nansum(area_matrix[flow_mask], axis=0)

        # Correct waterdepth for lakes
        waterdepth = waterdepth + plassen_depth_correction
        waterdepth = waterdepth[waterdepth >= 0]

        # Correct waterdepth for deep pools (volume below deepest point in centre
        # should not be considered to be conveyance)
        pools_id = [
            i[0] for i in np.argwhere(waterdepth.to_numpy()[:, 0] > centre_depth[0])
        ]
        for pool in pools_id:
            amount_deeper = waterdepth.iloc[pool, 0] - centre_depth[0]
            waterdepth.iloc[pool] -= amount_deeper

        # waterdepth[waterdepth > centre_depth[0]] = centre_depth[0]

        # Compute 2D volume as sum of area times depth
        self._fm_total_volume = np.array(
            np.nansum(
                area_matrix[wet_not_plas_mask] * waterdepth[wet_not_plas_mask], axis=0
            )
        )
        self._fm_flow_volume = np.array(
            np.nansum(area_matrix[flow_mask] * waterdepth[flow_mask], axis=0)
        )

        # For roughness we will need the original z-levels,
        # since geometry z will change below
        self._css_z_roughness = centre_level

        # Check for monotonicity (water levels should rise)
        mono_mask = self._check_monotonicity(centre_level, method=1)
        centre_level = centre_level[mono_mask]
        self._fm_total_volume = self._fm_total_volume[mono_mask]
        self._fm_flow_volume = self._fm_flow_volume[mono_mask]
        self._fm_wet_area = self._fm_wet_area[mono_mask]
        self._fm_flow_area = self._fm_flow_area[mono_mask]

        # Compute geometry above z0 - Water level dependent calculation
        self.set_logger_message("Computing cross-section from water levels")
        self._compute_css_above_z0(centre_level)

        # Compute geometry below z0 - Water level independent calculation
        self.set_logger_message("Computing cross-section from bed levels")
        self._extend_css_below_z0(
            centre_level,
            centre_depth,
            waterlevel,
            bedlevel_matrix,
            area_matrix,
            wet_not_plas_mask.iloc[:, 0].values,
        )

        # Compute 1D volume as integral of width with respect to z times length
        self._css_total_volume = np.append(
            [0], cumtrapz(self._css_total_width, self._css_z) * self.length
        )
        self._css_flow_volume = np.append(
            [0], cumtrapz(self._css_flow_width, self._css_z) * self.length
        )

        # If sd correction is run, these attributes will be updated.
        self._css_total_volume_corrected = self._css_total_volume
        self._css_flow_volume_corrected = self._css_flow_volume

        # convert to float64 array for uniformity
        # (apparently entries can be float32)
        self._css_z = np.array(self._css_z, dtype=np.dtype("float64"))

        fm_data["islake"] = plassen_mask

        # generate all mask points for the given cross_section
        # self.set_mask_output_list(fm_data, plassen_mask)

    def check_requirements(self):
        """
        Performs check on cross-section such that it
        hold up to requirements.
        """
        # Remove multiple zeroes in the bottom of the cross-section
        self._css_index_of_first_nonzero = self._check_remove_duplicate_zeroes()
        self._css_total_width = self._check_remove_zero_widths(self._css_total_width)
        self._css_flow_width = self._check_remove_zero_widths(self._css_flow_width)

        # Check if cross-sections are in increasing order
        self._css_z = self._check_increasing_order(self._css_z)
        self._css_total_width = self._check_increasing_order(self._css_total_width)
        self._css_flow_width = self._check_increasing_order(self._css_flow_width)

        # Check if total width is larger than flow width
        self._check_total_width_greater_than_flow_width()

    def calculate_correction(self) -> None:
        """
        This method computes a volume correction that cannot be captured within the
        constraints of 1D cross-sectional geometry, which forces an increasing width
        at increasing elevation and cannot deal with varying water levels. In reality,
        compartimentation of floodplains can cause a sudden
        increase of volume while the water level does not increase.

        Technically, the derivative of the volume over elevation (dV/dz) must be
        monotonically increasing, meaning it cannot increase slower than it did
        at a lower z. However, this is exactly what happens with compartimentalisation,
        see (see :term:`Summerdikes`).

        This method computes input for the 'summerdike' option in SOBEK, which requires three variables:

        - The crest level of the summerdike
        - The total volume behind the summerdike
        - The total volume behind the summerdike
        - The :ref:`transition height <parameter_sdtransitionheight>`
          over which the volumes are added to the geometry. This variable is set globally.

        We refer to the SOBEK technical manual for details on the implementation.

        The algorithm works as follows:

        1. The initial attributes (crest and volumes) are set and outputted (in debug mode)
        2. Optimise the attributes by minimising the cost function. For the cost function we
           use the squared relative error. The parameter :ref:`SDOptimisationMethod<parameter_sdoptimisationmethod>`
           is used to determine which output variables are used to compute the error.
        3. Add the extra volume to the css volume given the water level (see note below)

        .. note::
            In SOBEK, the extra volume is released following a polynomial function. In |project|, we approximate
            this with the following logistic function:

            :math:`C(h_k)=\Xi(1+e^{\log(\delta)\\tau^{-1}(h_k-(\gamma+\\tau/2))})^{-1}`

            where :math:`\Xi` is the volume correction (m3/s), :math:`\\tau` is the transition height,
            :math:`\delta` is an accuracy parameter, :math:`\gamma` is the crest level, and :math:`C(h_k)`
            is the added volume given water level :math:`h_k`. The default value of :math:`\delta` is
            0.00001, and not configurable.

        """

        # Set initial values for optimisation of parameters
        initial_total_error = self._css_total_volume - self._fm_total_volume
        initial_flow_error = self._css_flow_volume - self._fm_flow_volume

        initial_crest = self._css_z[np.nanargmin(initial_total_error)]
        initial_total_volume = np.abs(initial_total_error[-1])
        initial_flow_volume = np.abs(initial_flow_error[-1])

        self.set_logger_message(
            "Initial crest: {:.4f} m".format(initial_crest), level="debug"
        )
        self.set_logger_message(
            "Initial extra total area: {:.4f} m2".format(
                initial_total_volume / self.length
            ),
            level="debug",
        )
        self.set_logger_message(
            "Initial extra flow area: {:.4f} m2".format(
                initial_flow_volume / self.length
            ),
            level="debug",
        )

        # Optimise attributes
        opt = self._optimize_sd_storage(
            initial_crest=initial_crest,
            initial_total_volume=initial_total_volume,
            initial_flow_volume=initial_flow_volume,
        )

        # Unpack optimisation results
        transition_height = self.get_parameter(self.__cs_parameter_transitionheight_sd)
        crest_level = opt.get("crest_level")
        extra_total_volume = opt.get("extra_total_volume")
        extra_flow_volume = opt.get("extra_flow_volume")

        self.set_logger_message(
            "final costs: {:.2f}".format(opt.get("final_cost")), level="debug"
        )
        self.set_logger_message(
            "Optimizer msg: {}".format(opt.get("message")), level="debug"
        )
        self.set_logger_message(
            "Final crest: {:.4f} m".format(crest_level), level="debug"
        )
        self.set_logger_message(
            "Final total area: {:.4f} m2".format(extra_total_volume / self.length),
            level="debug",
        )
        self.set_logger_message(
            "Final flow area: {:.4f} m2".format(extra_flow_volume / self.length),
            level="debug",
        )

        extra_area_percentage = self.__get_extra_total_area(
            self._css_z, crest_level, transition_height
        )

        # Write to attributes
        self._css_total_volume_corrected = (
            self._css_total_volume + extra_area_percentage * extra_total_volume
        )
        self._css_flow_volume_corrected = (
            self._css_flow_volume + extra_area_percentage * extra_flow_volume
        )
        self.crest_level = crest_level
        self.transition_height = transition_height
        self.extra_total_volume = extra_total_volume
        self.extra_flow_volume = extra_flow_volume
        self.extra_total_area = extra_total_volume / self.length
        self.extra_flow_area = extra_flow_volume / self.length
        self.extra_area_percentage = extra_area_percentage
        self._css_is_corrected = True

    def assign_roughness(self) -> None:
        """
        This function builds a table of Chezy values as function of water level
        The roughnes is divides into two sections on the assumption of
        an alluvial (smooth) and nonalluvial (rough) part of the total
        cross-section. This division is made based on the final timestep.
        """

        # Compute roughness tabels
        self.set_logger_message("Building roughness table", "debug")
        self._build_roughness_tables()
        # Compute roughness widths
        self.set_logger_message("Computing section widths", "debug")
        self._compute_section_widths()
        # Compute floodplain base level
        self.set_logger_message("Computing base level", "debug")
        self._compute_floodplain_base()
        # done

    def get_number_of_faces(self) -> int:
        """use this method to return the number of 2D faces within control volume"""
        return len(self._fm_data.get("x"))

    def get_number_of_vertices(self) -> int:
        """Use this method to return the current number of geometry vertices"""
        return len(self._css_total_width)

    def reduce_points(
        self,
        count_after: int = 20,
        method: str = "visvalingam_whyatt",
        verbose: bool = True,
    ) -> None:
        """
        The cross-section geometry generated by |project| contains one point per output
        timestep in the 2D map file. This resolution is often too high given the
        complexity of the cross-sections, and results in very large input files for the
        1D model. Therefore |project| includes a simplification algorithm that reduces
        the number of points while preservering the shape of the geometry. This algorithm
        reduces as many points until the number specified in
        :ref:`MaximumPointsInProfile<parameter_maximumpointsinprofile>` is reached.
        We use the Visvalingam-Whyatt method of poly-line vertex reduction [VW]_.
        The :term:`Total width` is leading for the simplification of the geometry meaning
        that the choice for which points to remove to simplify the geometry is based on
        the total width. Subsequently, the corresponding point are removed from the :term:`Flow width`.

        .. [VW] Visvalingam, M and Whyatt J D (1993) "Line Generalisation by Repeated Elimination of Points", Cartographic J., 30 (1), 46 - 51 URL: http://web.archive.org/web/20100428020453/http://www2.dcs.hull.ac.uk/CISRG/publications/DPs/DP10/DP10.html
                Implemented vertex reduction methods:

        Args:
            n (int): asdf
            method (str): asdf
            verbose (bool): asdf

        Returns:
            None: This method directly works on the object attributes
        """

        n_before_reduction = self.get_number_of_vertices()

        points = np.array(
            [
                [self._css_z[i], self._css_total_width[i]]
                for i in range(n_before_reduction)
            ]
        )

        # The number of points is equal to n, it cannot be further reduced
        reduced_index = np.array([True] * n_before_reduction)

        if n_before_reduction > count_after:
            # default is the same value as it came
            if method.lower() == "visvalingam_whyatt":
                try:
                    simplifier = PS.VWSimplifier(points)
                    reduced_index = simplifier.from_number_index(count_after)
                except Exception as e:
                    self.set_logger_message(
                        "Exception thrown while using polysimplify: "
                        + "{}".format(str(e)),
                        "error",
                    )

        # Write to attributes
        self.z = self._css_z[reduced_index]
        self.total_width = self._css_total_width[reduced_index]
        self.flow_width = self._css_flow_width[reduced_index]

        self.set_logger_message(
            "Cross-section reduced "
            + "from {} ".format(n_before_reduction)
            + "to {} points".format(len(self.total_width))
        )

        self._css_is_reduced = True

    def set_face_output_list(self):
        """
        Generates a list of output mask points based on
        their values in the mask.

        writes to self.__output_mask_list

        Paramters:
            fm_data {dict} -- Dictionary containing x,y values.
            mask_array {NP.array} -- Array of values.
        """
        fm_data = self._fm_data

        # Properties keys
        cross_section_id_key = "cross_section_id"
        cross_section_region_key = "region"
        is_lake_key = "is_lake"
        bedlevel_key = "bedlevel"
        section_key = "section"
        region_key = "region"

        try:
            # Normalize np arrays to list for correct access
            x_coords = fm_data.get("x").tolist()
            y_coords = fm_data.get("y").tolist()
            region_list = fm_data.get("region").tolist()
            section_list = fm_data.get("section").tolist()
            bedlevel_list = fm_data.get("bedlevel").tolist()
            is_lake_mask_list = fm_data.get("islake").tolist()

            # Assume same length for x and y coords.
            for i in range(len(x_coords)):
                mask_properties = {
                    cross_section_id_key: self.name,
                    is_lake_key: is_lake_mask_list[i],
                    cross_section_region_key: region_list[i],
                    bedlevel_key: bedlevel_list[i],
                    region_key: region_list[i],
                    section_key: section_list[i],
                }
                mask_coords = (x_coords[i], y_coords[i])
                # Create the actual geojson element.
                output_mask = MaskOutputFile.create_mask_point(
                    mask_coords, mask_properties
                )

                if output_mask.is_valid:
                    self.__output_face_list.append(output_mask)
                    # self.set_logger_message(
                    #    'Added output mask at {} '.format(mask_coords) +
                    #    'for Cross Section {}.'.format(self.name),
                    #     level='debug')
                else:
                    self.set_logger_message(
                        "Invalid output mask at {} ".format(mask_coords)
                        + "for Cross Section {}, not added. ".format(self.name)
                        + "Reason {}".format(output_mask.errors()),
                        level="error",
                    )
        except Exception as e_error:
            self.set_logger_message(
                "Error setting output masks "
                + "for Cross Section {}. ".format(self.name)
                + "Reason: {}".format(str(e_error)),
                level="error",
            )

    def set_edge_output_list(self):
        """
        Generates a list of output mask points based on
        their values in the mask.

        writes to self.__output_mask_list

        Parameters:
            fm_data {dict} -- Dictionary containing x,y values.
            mask_array {NP.array} -- Array of values.
        """
        fm_data = self._fm_data

        # Properties keys
        cross_section_id_key = "cross_section_id"
        cross_section_region_key = "region"
        roughness_section_key = "section"

        try:
            # Normalize np arrays to list for correct access
            x_coords = fm_data.get("edge_x").tolist()
            y_coords = fm_data.get("edge_y").tolist()
            section_list = fm_data.get("edge_section").tolist()
            # Assume same length for x and y coords.
            for i in range(len(x_coords)):
                mask_properties = {
                    cross_section_id_key: self.name,
                    roughness_section_key: section_list[i],
                }
                mask_coords = (x_coords[i], y_coords[i])
                # Create the actual geojson element.
                output_mask = MaskOutputFile.create_mask_point(
                    mask_coords, mask_properties
                )

                if output_mask.is_valid:
                    self.__output_edge_list.append(output_mask)
                else:
                    self.set_logger_message(
                        "Invalid output mask at {} ".format(mask_coords)
                        + "for Cross Section {}, not added. ".format(self.name)
                        + "Reason {}".format(output_mask.errors()),
                        level="error",
                    )
        except Exception as e_error:
            self.set_logger_message(
                "Error setting output masks "
                + "for Cross Section {}. ".format(self.name)
                + "Reason: {}".format(str(e_error)),
                level="error",
            )

    def interpolate_roughness_table(self, tablename, z_values):
        """
        Interpolates the roughness table to z values
        """
        table_name_list = ["main", "floodplain1"]

        if tablename not in table_name_list:
            raise KeyError("tablename not in list {}".format(table_name_list))

        pass

    def _check_remove_duplicate_zeroes(self):
        """
        Removes duplicate zeroes in the total width
        """
        mask = np.array([True] * len(self._css_z))

        # Remove multiple 0s in the total width
        index_of_first_nonzero = max(1, np.argwhere(self._css_total_width != 0)[0][0])

        return index_of_first_nonzero

        # self._css_z = self._return_first_item_and_after_index(self._css_z, index_of_first_nonzero)
        # self._css_flow_width = self._return_first_item_and_after_index(self._css_flow_width, index_of_first_nonzero)
        # self._css_total_width = self._return_first_item_and_after_index(self._css_total_width, index_of_first_nonzero)
        # self.set_logger_message(f'Removed {index_of_first_nonzero-1} duplicate zero widths', 'debug')

    @staticmethod
    def _return_first_item_and_after_index(listin, after_index):
        return np.append(listin[0], listin[after_index:].tolist())

    def _check_remove_zero_widths(self, width_array):
        """
        A zero width may lead to numerical instability
        """

        minwidth = self.get_parameter(self.__cs_parameter_minwidth)
        width_array[width_array < minwidth] = minwidth

        return width_array

    def _combined_optimisation_func(self, opt_in):
        """
        Cost function, combines total volume error and flow volume error
        """
        (crest_level, extra_total_volume, extra_flow_volume) = opt_in
        transition_height = self.get_parameter(self.__cs_parameter_transitionheight_sd)

        predicted_total_volume = (
            self._css_total_volume
            + self.__get_extra_total_area(self._css_z, crest_level, transition_height)
            * extra_total_volume
        )

        predicted_flow_volume = (
            self._css_flow_volume
            + self.__get_extra_total_area(self._css_z, crest_level, transition_height)
            * extra_flow_volume
        )

        return self.__return_volume_error(
            predicted_total_volume + predicted_flow_volume,
            self._fm_total_volume + self._fm_flow_volume,
        )

    def _optimisation_func(self, opt_in, *args):
        """
        Objective function used in optimising a delta-h correction
        for parameters:
            crest_level         : level at which the correction begins
            transition_height   : height over which volume is released
            extra_volume        : total extra volume


        :param opt_in: tuple
        :return:
        """
        if args[0][0] == "both":
            (crest_level, extra_volume) = opt_in

        else:
            (extra_volume) = opt_in
            crest_level = args[0][2]

        volume = args[0][1]
        transition_height = self.get_parameter(self.__cs_parameter_transitionheight_sd)

        predicted_volume = (
            volume
            + self.__get_extra_total_area(self._css_z, crest_level, transition_height)
            * extra_volume
        )
        return self.__return_volume_error(predicted_volume, self._fm_total_volume)

    def _optimize_sd_storage(
        self, initial_crest, initial_total_volume, initial_flow_volume
    ) -> dict:
        """
        Optimised the crest level and volumes

        Returns:
            Dictionary with optimised values, final cost and optimisation message
        """
        # Default option
        sdoptimisationmethod = self.get_parameter(
            self.__cs_parameter_sdoptimisationmethod
        )
        if sdoptimisationmethod not in [0, 1, 2]:
            # this should be handled in inifile instead
            self.set_logger_message(
                "sdoptimisationmethod is {} but should be 0, 1, or 2. Defaulting to 0".format(
                    sdoptimisationmethod
                ),
                level="warning",
            )
            sdoptimisationmethod = 0

        if sdoptimisationmethod == 0:
            self.set_logger_message("Optimising SD on total volume", level="debug")

            # Optimise crest on total volume
            opt = so.minimize(
                self._optimisation_func,
                (initial_crest, initial_total_volume),
                args=["both", self._css_total_volume],
                method="Nelder-Mead",
                tol=1e-6,
            )

            crest_level = opt["x"][0]
            extra_total_volume = np.max([opt["x"][1], 0])

            # Optimise flow volume
            opt2 = so.minimize(
                self._optimisation_func,
                (initial_flow_volume),
                args=["notboth", self._css_flow_volume, crest_level],
                method="Nelder-Mead",
                tol=1e-6,
            )
            extra_flow_volume = np.min([np.max([opt2["x"][0], 0]), extra_total_volume])

        elif self.get_parameter(self.__cs_parameter_sdoptimisationmethod) == 1:
            self.set_logger_message("Optimising SD on flow volume", level="debug")

            # Optimise crest on flow volume
            opt = so.minimize(
                self._optimisation_func,
                (initial_crest, initial_total_volume),
                args=["both", self._css_flow_volume],
                method="Nelder-Mead",
                tol=1e-6,
            )
            crest_level = opt["x"][0]
            extra_flow_volume = np.max([opt["x"][1], 0])

            # Optimise total volume
            opt2 = so.minimize(
                self._optimisation_func,
                (initial_flow_volume),
                args=["notboth", self._css_total_volume, crest_level],
                method="Nelder-Mead",
                tol=1e-6,
            )
            extra_total_volume = np.max([np.max([opt2["x"][0], 0]), extra_flow_volume])

        elif self.get_parameter(self.__cs_parameter_sdoptimisationmethod) == 2:
            self.set_logger_message(
                "Optimising SD on both flow and total volumes", level="debug"
            )
            opt = so.minimize(
                self._combined_optimisation_func,
                (initial_crest, initial_total_volume, initial_flow_volume),
                method="Nelder-Mead",
                tol=1e-6,
            )

            crest_level = opt["x"][0]
            extra_total_volume = np.max([opt["x"][1], 0])
            extra_flow_volume = np.min([np.max([opt["x"][2], 0]), extra_total_volume])

        return {
            "crest_level": crest_level,
            "extra_total_volume": extra_total_volume,
            "extra_flow_volume": extra_flow_volume,
            "final_cost": opt["fun"],
            "message": opt["message"],
        }

    def _check_increasing_order(self, list_points):
        """runs"""
        for i in range(1, len(list_points)):
            if list_points[i] <= list_points[i - 1]:
                list_points[i] = list_points[i - 1] + 0.001
        return list_points

    def _build_roughness_tables(self):

        # Find roughness tables for each section
        chezy_fm = self._fm_data.get("chezy").iloc[
            :, self.get_parameter(self.__cs_parameter_skip_maps) :
        ]

        sections = np.unique(self._fm_data.get("edge_section"))

        for section in sections:
            chezy_section = chezy_fm[self._fm_data["edge_section"] == section]
            if self.get_parameter(self.__cs_parameter_Frictionweighing) == 0:
                friction = self._friction_weighing_simple(chezy_section, section)
            elif self.get_parameter(self.__cs_parameter_Frictionweighing) == 1:
                friction = self._friction_weighing_area(chezy_section, section)
            else:
                raise ValueError(
                    "unknown option for roughness weighing: {}".format(
                        self.get_parameter(self.__cs_parameter_Frictionweighing)
                    )
                )

            self.friction_tables[self._section_map[str(section)]] = FrictionTable(
                level=self._css_z_roughness, friction=friction
            )

    def _friction_weighing_simple(self, link_chezy, section):
        """Simple mean, no weight"""
        # Remove chezy where zero
        link_chezy = link_chezy.replace(0, np.NaN)
        output = link_chezy.mean(axis=0).replace(np.NaN, 0)

        return output.values

    def _friction_weighing_area(self, link_chezy, section):
        """
        Compute chezy by weighted average. Weights are determined based on area.

        Friction values are known at flow links, while areas are known at flow faces.

        The area of a flow link is defined as the average of the two faces it connects.

        """
        # Remove chezy where zero
        link_chezy = link_chezy.replace(0, np.NaN)
        # efs are the two faces the edge connects to
        efs = self._fm_data["edge_faces"][self._fm_data["edge_section"] == section]
        link_area = []
        for ef in efs:
            # compute the mean area for the two connecting faces
            link_area.append(self._fm_data.get("area_full").reindex(ef).mean())

        # the weight of one link is defined as the sum of the linked areas
        link_weight = link_area / np.sum(link_area)

        output = np.sum(link_chezy.values.T * link_weight, axis=1)
        output[np.isnan(output)] = 0
        return output

    def _compute_section_widths(self) -> None:
        """
        Computes sections widths by dividing the area assigned to a section
        by the length of the cross-section.

        If the sum of the section widths is smaller than the flow width, the
        width is increase proportionally
        """
        total_area = sum(self._fm_data["area"])
        unassigned_area = sum(self._fm_data["area"][self._fm_data["section"] == -999])
        if unassigned_area > 0:
            self.set_logger_message(
                f"{unassigned_area} m2 was not assigned to any section in input files, and is added to the main section",
                "warning",
            )

        for section in [1, 2, 3]:
            if section == 1:
                section_area = (
                    np.sum(self._fm_data["area"][self._fm_data["section"] == section])
                    + unassigned_area
                ) / self.length
            else:
                section_area = (
                    np.sum(self._fm_data["area"][self._fm_data["section"] == section])
                    / self.length
                )
            self.section_widths[self._section_map[str(section)]] = section_area

        # Finally, the sum of section width should be greater or equal to the flow width
        self._check_section_widths_greater_than_flow_width()
        self._check_section_widths_greater_than_minimum_width()

    def _compute_floodplain_base(self) -> None:
        """
        Sets the self.floodplain_base attribute. The floodplain
        will be set at least 0.5 meter below the crest of the
        embankment, and otherwise at the average hight of the floodplain
        """
        tolerance = self.get_inifile().get_parameter("sdfloodplainbase")
        # Mean bed level in section 2 (floodplain)
        floodplain_mask = self._fm_data.get("section") == 2
        if floodplain_mask.sum():
            mean_floodplain_elevation = np.nanmean(
                self._fm_data["bedlevel"][floodplain_mask]
            )

            # Tolerance. Base level must at least be some below the crest to prevent
            # numerical issues

            if (self.crest_level - tolerance) < mean_floodplain_elevation:
                self.floodplain_base = self.crest_level - tolerance
                self.set_logger_message(
                    f"Mean floodpl. elev. ({mean_floodplain_elevation:.2f} m)"
                    + f"higher than crest level ({self.crest_level:.2f}) + "
                    + f"tolerance ({tolerance} m)",
                    "warning",
                )
            else:
                self.floodplain_base = mean_floodplain_elevation
                self.set_logger_message(
                    f"Floodplain base level set to {mean_floodplain_elevation:.2f} m",
                    "debug",
                )
        else:
            self.floodplain_base = self.crest_level - tolerance
            self.set_logger_message(
                f"No Floodplain found, floodplain defaults to {self.crest_level - tolerance}"
            )

    def _calc_chezy(self, depth, manning):
        return depth ** (1 / float(6)) / manning

    def _identify_lakes(self, waterdepth):
        """
        This algorithms determines whether a :ref:`2D cell <section_parsing_2d_data>` should
        be marked as a :term:`Lake<Lakes>`.

        Cells are marked as lake if the following conditions are both met:
            - the waterdepth on timestep :ref:`LakeTimeSteps <parameter_laketimesteps>` is positive
            - the waterdepth on timestep :ref:`LakeTimeSteps <parameter_laketimesteps>` is at least 1 cm higher than the waterlevel on timestep 0.

        Next, the following steps are taken

        - It is determined at what timestep the waterlevel in the lake starts rising. From that point on the lake counts as regular geometry and counts toward the total volume. A cell is considered active if its waterlevel has risen by 1 mm.
        - A correction matrix is built that contains the 'lake water level' for each lake cell. This matrix is subtracted from the waterdepth to compute volumes.


        Args:
            waterdepth (pandas dataframe):

        Returns:
            lake_mask (ndarray): mask of all cells that are a 'lake'
            wet_not_lake_mask (ndarray): mask of all cells that are wet, but not a lake
            lake_depth_correction (ndarray): the depth of a lake at the start of the 2D computation

        """
        # preallocate arrays
        plassen_depth_correction = np.zeros(waterdepth.shape, dtype=float)

        # check for non-rising waterlevels
        waterdepth_diff = np.diff(waterdepth, n=1, axis=-1)

        # find all wet cells
        wet_mask = waterdepth > 0

        # find all lakes
        lake_mask = (
            waterdepth.T.iloc[self.get_parameter(self.__cs_parameter_plassen_timesteps)]
            > 0
        ) & (
            np.abs(
                waterdepth.T.iloc[
                    self.get_parameter(self.__cs_parameter_plassen_timesteps)
                ]
                - waterdepth.T.iloc[0]
            )
            <= 0.01
        )

        self.plassen_mask = lake_mask

        # Plassen_mask_time is to determine at whata timestep the lake starts rising again.
        plassen_mask_time = np.zeros((len(waterdepth.T), len(lake_mask)), dtype=bool)

        # At t=0, all lakes are inactive
        plassen_mask_time[0, :] = lake_mask

        # walk through dataframe in time, for each timestep check
        # when to unmask a plassen cell
        for i, diff in enumerate(waterdepth_diff.T):
            final_mask = reduce(
                np.logical_and, [(diff <= 0.001), (plassen_mask_time[i] == True)]
            )
            plassen_mask_time[i + 1, :] = final_mask

        plassen_mask_time = pd.DataFrame(plassen_mask_time).T

        # The depth of a lake is the waterdepth at t=0
        for i, depths in enumerate(waterdepth):
            plassen_depth_correction[lake_mask, i] = -waterdepth.T.iloc[0][lake_mask]

        # correct wet cells for plassen
        wet_not_plas_mask = reduce(
            np.logical_and, [(wet_mask == True), (plassen_mask_time == False)]
        )

        return lake_mask, wet_not_plas_mask, plassen_depth_correction

    def _compute_css_above_z0(self, centre_level) -> None:
        """
        This method computes for each level (z) above the water level at the first 2D output (z0),
        the corresponding :term:`Total width` and :term:`Flow width`. This is done in the following way:

        1. compute the total width by dividing the wet area by the cross-section length
        2. compute the flow width by dividing the flow area by the cross-section length
        3. Correct the flow width such that flow width is always increasing.

        Args:
            centre_level: the water level at the cross-section location (x, y), which is typically at the centre of the control volume

        Return:

            None: this method writes to the cross-section attributes _css_z, _css_total_width and _css_flow_width
        """

        # Set the level
        self._css_z = centre_level

        # Set nan's to the minimum. NaN values occur lower z values at the start of computation
        self._css_z[np.isnan(self._css_z)] = np.nanmin(self._css_z)

        # Compute widths
        self._css_total_width = np.array(self._fm_wet_area) / self.length
        self._css_flow_width = np.array(self._fm_flow_area) / self.length

        # Flow width must increase at each z
        for i in range(2, len(self._css_flow_width) + 1):
            self._css_flow_width[-i] = np.min(
                [self._css_flow_width[-i], self._css_flow_width[-i + 1]]
            )

    def _distinguish_flow_from_storage_2(self, waterdepth, velocity):
        """
        This method determines which cells should be considered 'flowing'. A cell is considered flowing if all following conditions are met:

        - the waterdepth greater than 0
        - the velocity is greater than :ref:`AbsoluteVelocityThreshold<parameter_absolutevelocitythreshold>`
        - the velocity is greater than the product of the mean velocity (of all cells within the control volume) and the :ref:`RelativeVelocityThreshold<parameter_relativevelocitythreshold>`

        Args:
            waterdepth (pandas Dataframe) with cell id
                for index and time in columns
            velocity (pandas Dataframe() with cell id
                for index and time in columns

        Returns:
            flow_mask (pandas Dataframe) with cell id for index and time in columns. True for flow, False for storage
        """
        flow_mask = (
            (waterdepth > 0)
            & (velocity > self.get_parameter(self.__cs_parameter_velocity_threshold))
            & (
                velocity
                > self.get_parameter(self.__cs_parameter_relative_threshold)
                * np.mean(velocity)
            )
        )

        # shallow flows should not be used for identifying storage
        # (cells with small velocities are uncorrectly seen as storage)
        waterdepth_correction = (waterdepth > 0) & (
            waterdepth < self.get_parameter(self.__cs_parameter_min_depth_storage)
        )

        # combine flow and depth mask to avoid assigning shallow flows
        flow_mask = flow_mask | waterdepth_correction

        return flow_mask

    def _distinguish_flow_from_storage(self, waterdepth, velocity):
        """
        This method determines which cells should be considered 'flowing'. A cell is considered flowing if all following conditions are met:

        - the waterdepth greater than 0
        - the velocity is greater than :ref:`AbsoluteVelocityThreshold<parameter_absolutevelocitythreshold>`
        - the velocity is greater than the product of the mean velocity (of all cells within the control volume) and the :ref:`RelativeVelocityThreshold<parameter_relativevelocitythreshold>`

        Args:
            waterdepth (pandas Dataframe) with cell id
                for index and time in columns
            velocity (pandas Dataframe() with cell id
                for index and time in columns

        Returns:
            flow_mask (pandas Dataframe) with cell id for index and time in columns. True for flow, False for storage
        """
        flow_mask = (
            (waterdepth > 0)
            & (velocity > self.get_parameter(self.__cs_parameter_velocity_threshold))
            & (
                velocity
                > self.get_parameter(self.__cs_parameter_relative_threshold)
                * np.mean(velocity)
            )
        )

        # shallow flows should not be used for identifying storage
        # (cells with small velocities are uncorrectly seen as storage)
        waterdepth_correction = (waterdepth > 0) & (
            waterdepth < self.get_parameter(self.__cs_parameter_min_depth_storage)
        )

        # combine flow and depth mask to avoid assigning shallow flows
        # flow_mask = flow_mask | waterdepth_correction

        return flow_mask

    def _extend_css_below_z0(
        self,
        centre_level,
        centre_depth,
        waterlevel,
        bedlevel_matrix,
        area_matrix,
        wet_not_plas_mask,
    ) -> None:
        """
        This methods computeS for level (z) below the water level at the first 2D output (z0) the corresponding
        :term:`Total width` and :term:`Flow width`. This is done in the following way:

        1. Take a number of steps (see note below) from z at t0 do the bed level at the :term:`Cross-section location`
        2. for each step, determine which cells should be counted
            - the bed level of the cell should be higher than the water level plus the tolerance (see note below)
            - the cell should not be part of a :term:`Lakes`
        3. Since there is no information on flow velocities, we cannot determine which cells are flowing and width are storage.
           Therefore is is decided like this
            - If :ref:`ExtrapolateStorage <parameter_extrapolatestorage>` is True, the flow area is the minimum of the flow area at t0 (from :ref:`wl_dependent_css`)
            - if :ref:`ExtrapolateStorage <parameter_extrapolatestorage>` is False, the flow area is equal to the total area

        .. note::
            - the number of steps between z at t0 to bed level is hard-coded at 10.
            - the tolerance for deciding which cell is wet is hardcoded at -1e-3


        Returns:
            None: This method extends the following attributes:
                    _css_z
                    _css_total_width
                    _css_flow_width
                    _fm_wet_area
                    _fm_flow_area
                    _fm_flow_volume
                    _fm_total_volume
        """

        bedlevel = self._fm_data.get("bedlevel").values
        cell_area = self._fm_data.get("area").values
        flow_area_at_z0 = self._fm_flow_area[0]
        lowest_level_of_css = (
            centre_level[0] - centre_depth[0]
        )  # this is in fact the bed level at centre point
        centre_level_at_t0 = centre_level[0]
        waterlevel_at_t0 = waterlevel.values[:, 0]
        waterdepth_at_t0 = waterlevel_at_t0 - bedlevel
        waterdepth_at_t0[np.isnan(waterdepth_at_t0)] = 0
        tolerance = -1e-3  # at last point, this is still considered wet.

        # Take steps from z0 downward to the lowest level
        for dz in np.linspace(0, centre_level_at_t0 - lowest_level_of_css, 10):
            centre_level_at_dz = centre_level_at_t0 - dz
            total_wet_area = np.nansum(
                cell_area[((waterdepth_at_t0 - dz) > tolerance) & wet_not_plas_mask]
            )

            # Extension of flow/storage below z0
            if not self.get_parameter("extrapolatestorage"):
                total_flow_area = total_wet_area
            elif self.get_parameter("extrapolatestorage"):
                total_flow_area = np.min([total_wet_area, flow_area_at_z0])

            self._css_z = CrossSection.__append_to_start(
                self._css_z, centre_level_at_dz
            )
            self._css_total_width = CrossSection.__append_to_start(
                self._css_total_width, total_wet_area / self.length
            )
            self._css_flow_width = CrossSection.__append_to_start(
                self._css_flow_width, total_flow_area / self.length
            )
            self._fm_wet_area = CrossSection.__append_to_start(
                self._fm_wet_area, total_wet_area
            )
            self._fm_flow_area = CrossSection.__append_to_start(
                self._fm_flow_area, total_flow_area
            )
            self._fm_flow_volume = np.insert(self._fm_flow_volume, 0, np.nan)
            self._fm_total_volume = np.insert(self._fm_total_volume, 0, np.nan)

    @staticmethod
    def __get_extra_total_area(
        waterlevel, crest_level, transition_height: float, hysteresis=False
    ):
        """
        releases extra area dependent on waterlevel using a logistic (sigmoid) function
        """
        delta = 0.00001  # accuracy parameter
        return 1 / (
            1
            + np.e
            ** (
                np.log(delta)
                / (transition_height)
                * (waterlevel - (crest_level + 0.5 * transition_height))
            )
        )

    @staticmethod
    def __append_to_start(array, to_add):
        """
        adds ``to add`` to beginning of array
        """
        return np.insert(array, 0, to_add)

    def __extend_css_with_constant_waterdepth(
        self, centre_level, centre_depth, bedlevel_matrix, area_matrix, plassen_mask
    ):
        """
        Extends the cross-sectional information below the water level
        at the first timestep,
        under assumption that the polygon formed by the water level
        and the bed level is convex.
        It works by walking down to the bed level at the center point
        in 'virtual water level steps'.
        At each step, we sum the area of cells width bedlevels which
        would be submerged at that virtual water level.
        """
        filter_by_depth_percentage = (
            self.get_parameter(self.__cs_parameter_bedlevelcriterium) * 100
        )
        filter_value = FE.empirical_ppf(
            [filter_by_depth_percentage], bedlevel_matrix.values.T[0]
        )[0]
        self.set_logger_message(
            "Lowest {}% of bed levels are filtered (z<{:.4f}m)".format(
                filter_by_depth_percentage, filter_value
            ),
            level="debug",
        )
        level_z0 = centre_level[0]
        bdata = bedlevel_matrix[~plassen_mask]
        bmask = (bdata < level_z0) & (bdata >= filter_value)

        self.set_logger_message(
            "Number of points below z0 after applying filter: {}".format(
                np.sum(bmask.values.T[0])
            ),
            level="debug",
        )

        # Compute values at z0
        bedlevels_below_z0 = bdata[bmask]
        lowest_level_below_z0 = centre_level[0] - centre_depth[0]
        flow_area_at_z0 = self._fm_flow_area[0]
        total_area_at_z0 = self._fm_wet_area[0]

        for unique_level_below_z0 in reversed(
            np.linspace(lowest_level_below_z0, level_z0, 10)
        ):

            # count area
            areas = area_matrix[bedlevels_below_z0 <= unique_level_below_z0]

            # Set area, such that the width computed from this area is equal or lower
            # than the minimum width from the flow-dependent level
            area_at_unique_level = np.min(
                [np.nansum(areas.values.T[-1]), total_area_at_z0]
            )

            # Extension of flow/storage below z0
            if self.get_parameter(self.__cs_parameter_storagemethod_wli) == 0:
                flow_area_at_unique_level = area_at_unique_level
            elif self.get_parameter(self.__cs_parameter_storagemethod_wli) == 1:
                flow_area_at_unique_level = np.min(
                    [area_at_unique_level, flow_area_at_z0]
                )

            # Insert values in existing arrays
            self._css_z = np.insert(self._css_z, 0, unique_level_below_z0)
            self._css_flow_width = np.insert(
                self._css_flow_width, 0, flow_area_at_unique_level / self.length
            )
            self._css_total_width = np.insert(
                self._css_total_width, 0, area_at_unique_level / self.length
            )
            self._fm_wet_area = np.insert(self._fm_wet_area, 0, area_at_unique_level)
            self._fm_flow_area = np.insert(
                self._fm_flow_area, 0, flow_area_at_unique_level
            )
            self._fm_flow_volume = np.insert(self._fm_flow_volume, 0, np.nan)
            self._fm_total_volume = np.insert(self._fm_total_volume, 0, np.nan)

    @staticmethod
    def __return_volume_error(predicted, measured, gof="rmse"):
        """
        Returns the squared relative error
        """
        non_nan_mask = ~np.isnan(predicted) & ~np.isnan(measured)
        predicted = predicted[non_nan_mask]
        measured = measured[non_nan_mask]
        error = np.array(predicted - measured) / np.maximum(
            np.array(measured), np.ones(len(measured))
        )
        return np.sum(error**2)

    @staticmethod
    def _check_monotonicity(arr, method=2):
        """
        for given input array, create mask such that when applied to the array,
        all values are monotonically rising

        method 1: remove values were z is falling from array
        method 2: sort array such that z is always rising (default)

        Arguments:
            arr: 1d numpy array

        return:
            mask such that arr[mask] is monotonically rising
        """
        if method == 1:
            mask = np.array([True])
            for i in range(1, len(arr)):
                # Last index that had rising value
                j = np.argwhere(mask)[-1][0]
                if arr[i] > arr[j]:
                    mask = np.append(mask, True)
                else:
                    mask = np.append(mask, False)

            return mask
        elif method == 2:
            return np.argsort(arr)

    def _check_total_width_greater_than_flow_width(self):
        """
        If total width is smaller than flow width, set flow width to total width
        """
        mask = self._css_flow_width > self._css_total_width
        self._css_flow_width[mask] = self._css_total_width[mask]
        self.set_logger_message(
            f"Reduces flow widths at {sum(mask)} points to be same as total", "debug"
        )

    def _check_section_widths_greater_than_flow_width(self):
        total_section_width = 0
        for key, width in self.section_widths.items():
            total_section_width += width

        dif = self.flow_width[-1] - total_section_width
        if dif > 0:
            self.section_widths["main"] += dif
            self.set_logger_message(
                f"Increased main section width by {dif:.2f} m", "warning"
            )

    def _check_section_widths_greater_than_minimum_width(self) -> bool:
        """
        Main section width must be greater than minimum profile width, or
        it is ignored by SOBEK 3
        """

        dif = self.section_widths["main"] - self._css_flow_width[0]

        # cm accuracy, and at least 10 cm difference
        # TODO: decide and implement some accuracy, e.g. 1e-3.
        # rounding errors may still lead to problems if main==flow width [0]
        # in sobek
        tol = 0.10
        dif = math.floor(dif * 100) / 100

        if (dif - tol) < 0:
            self.section_widths["main"] -= dif - tol
            self.section_widths["floodplain1"] += dif - tol
            self.set_logger_message(
                f"Increased main section width by {-1*(dif-tol):.2f}", "warning"
            )
            return True
        return False

    def get_parameter(self, key: str):
        return self.get_inifile().get_parameter(key)
