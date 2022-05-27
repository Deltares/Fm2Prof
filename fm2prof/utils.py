"""
Utilities for FM2PROF
"""

import os
import locale
import shutil

from typing import Tuple, Union, Optional, Dict, List, Generator
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import numpy as np
from fm2prof.common import FM2ProfBase
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

locale.setlocale(locale.LC_TIME, 'nl_NL.UTF-8')


font = {'family' : 'Bahnschrift',
            'weight' : 'normal',
            'size'   : 18}
mpl.rc('font', **font)

mpl.rcParams["axes.unicode_minus"] = False

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["k", "00cc96","#0d38e0"]*3,
                                            linestyle=["-"]*3+["--"]*3+['-.']*3,
                                            linewidth=np.linspace(0.5, 3, 9)) 


class GenerateCrossSectionLocationFile:
    def __init__(self, networkdefinitionfile:Union[str, Path], crossectionlocationfile:Union[str, Path], branchrulefile:Optional[Union[str, Path]]=None):
        """
        Builds a cross-section input file for FM2PROF from a DIMR network definition file.

        The distance between cross-section is computed from the differences between the offsets/chainages.
        The beginning and end point of each branch are treated as half-distance control volumes.

        Arguments
            networkdefinitionfile: path to the input file

            crossectionlocationfile: path to the output file

            branchrulefile: path to the branchrulefile
        """
        networkdefinitionfile, crossectionlocationfile, branchrulefile = map(Path, [networkdefinitionfile, crossectionlocationfile, branchrulefile])
        
        required_files = (networkdefinitionfile.is_file(), crossectionlocationfile.is_file())
        if not all(required_files): raise FileNotFoundError

        self._networkdeffile_to_input(networkdefinitionfile, crossectionlocationfile, branchrulefile)

    @staticmethod
    def parse_NetworkDefinitionFile(networkdefinitionfile:Path, branchrules:Optional[Dict]=None)->Dict:
        """
        Output:

            x,y : coordinates of cross-section
            cid : name of the cross-section
            cdis: half-way distance between cross-section points on either side
            bid : name of the branch
            coff:  chainage of cross-section on branch

        """
        if not branchrules: branchrules={}

        # Open network definition file, for each branch extract necessary info
        x = []  # x-coordinate of cross-section centre
        y = []  # y-coordinate of cross-section centre
        cid = []  # id of cross-section
        bid = []  # id of 1D branch
        coff = []  # offset of cross-section on 1D branch ('chainage')
        cdis = []  # distance of 1D branch influenced by crosss-section ('vaklengte')

        with open(networkdefinitionfile, "r") as f:
            for line in f:
                if line.strip().lower() == "[branch]":
                    branchid = f.readline().split("=")[1].strip()
                    xlength = 0
                    for i in range(10):
                        bline = f.readline().strip().lower().split("=")
                        if bline[0].strip() == "gridpointx":
                            xtmp = list(map(float, bline[1].split()))
                        elif bline[0].strip() == "gridpointy":
                            ytmp = list(map(float, bline[1].split()))
                        elif bline[0].strip() == "gridpointids":
                            cidtmp = bline[1].split(";")
                        elif bline[0].strip() == "gridpointoffsets":
                            cofftmp = list(map(float, bline[1].split()))

                            # compute distance between control volumes
                            cdistmp = np.append(np.diff(cofftmp) / 2, [0]) + np.append(
                                [0], np.diff(cofftmp) / 2
                            )

                    # Append branchids
                    bidtmp = [branchid] * len(xtmp)

                    # strip cross-section ids
                    cidtmp = [c.strip() for c in cidtmp]

                    # Correct end points (: at end of branch, gridpoints of this branch and previous branch
                    # occupy the same position, which does not go over well with fm2profs classification algo)
                    offset = 1
                    xtmp[0] = np.interp(offset, cofftmp, xtmp)
                    ytmp[0] = np.interp(offset, cofftmp, ytmp)
                    offset = cofftmp[-1] - 1
                    xtmp[-1] = np.interp(offset, cofftmp, xtmp)
                    ytmp[-1] = np.interp(offset, cofftmp, ytmp)

                    # Apply Branchrules
                    if branchid in branchrules:
                        rule = branchrules[branchid]
                        xtmp, ytmp, cidtmp, cdistmp, bidtmp, cofftmp = self._applyBranchRules(
                            rule, xtmp, ytmp, cidtmp, cdistmp, bidtmp, cofftmp
                        )

                        c = len(xtmp)
                        for ic in xtmp, ytmp, cidtmp, cdistmp, bidtmp, cofftmp:
                            if len(ic) != c:
                                print("koen")

                    # Append all points
                    x.extend(xtmp)
                    y.extend(ytmp)
                    cid.extend(cidtmp)
                    cdis.extend(cdistmp)
                    bid.extend(bidtmp)
                    coff.extend(cofftmp)

        return dict(x=x, y=y, css_id=cid, css_len=cdis, branch_id=bid, css_offset=coff)
    
    def _networkdeffile_to_input(self,
        networkdefinitionfile:Path, crossectionlocationfile:Path, branchrulefile:Optional[Path]=None):
        branchrules:dict = {}
        
        if branchrulefile: branchrules = self._parseBranchRuleFile(branchrulefile)
        
        network_dict = self.parse_NetworkDefinitionFile(
            networkdefinitionfile, branchrules
        )

        self._writeCrossSectionLocationFile(crossectionlocationfile, network_dict)


    def _applyBranchRules(rule, x, y, cid, cdis, bid, coff):
        bfunc = {
            "onlyedges": lambda x: [x[0], x[-1]],
            "ignoreedges": lambda x: x[1:-1],
            "ignorelast": lambda x: x[:-1],
            "ignorefirst": lambda x: x[1:],
        }

        disfunc = {
            "onlyedges": lambda x: [sum(x) / 2] * 2,
            "ignoreedges": lambda x: [sum(x[:2]), *x[2:-2], sum(x[-2:])],
            "ignorelast": lambda x: [*x[:-2], sum(x[-2:])],
            "ignorefirst": lambda x: [sum(x[:2]), *x[2:]],
        }

        try:
            bf = bfunc[rule.lower().strip()]
            df = disfunc[rule.lower().strip()]
            return bf(x), bf(y), bf(cid), df(cdis), bf(bid), bf(coff)
        except KeyError:
            raise NotImplementedError(rule)

    def _parseBranchRuleFile(branchrulefile:Path)->Dict:
        branchrules:dict = {}
        with open(branchrulefile, "r") as f:
            for line in f:
                key = line.split(",")[0].strip()
                value = line.split(",")[1].strip()

                branchrules[key] = value

        return branchrules

    def _writeCrossSectionLocationFile(crossectionlocationfile:Path, network_dict:Dict):
        """
        List inputs:

        x,y : coordinates of cross-section
        cid : name of the cross-section
        cdis: half-way distance between cross-section points on either side
        bid : name of the branch
        coff:  chainage of cross-section on branch
        """
        with open(crossectionlocationfile, "w") as f:
            f.write("name,x,y,length,branch,offset\n")
            for i in range(len(x)):
                f.write(
                    f"{cid[i]}, {x[i]:.4f}, {y[i]:.4f}, {cdis[i]:.2f}, {bid[i]}, {coff[i]:.2f}\n"
                )


class VisualiseOutput(FM2ProfBase):
    __cssdeffile = "CrossSectionDefinitions.ini"
    __volumefile = "volumes.csv"
    __rmainfile = "roughness-Main.ini"
    __rfp1file = "roughness-FloodPlain1.ini"

    def __init__(
        self, output_directory: str, figure_type: str = "png", overwrite: bool = True
    ):
        self._create_logger()
        self.output_dir = Path(output_directory)
        self.fig_dir = self._generate_output_dir()
        self._set_files()
        self._ref_geom_y = []
        self._ref_geom_tw = []
        self._ref_geom_fw = []

    def figure_roughness_longitudinal(self, branch:str):
        """
        Assumes the following naming convention:
        [branch]_[optional:branch_order]_[chainage]
        """
        
        fig, ax = plt.subplots(1, figsize=(12, 5))

        css = self.get_cross_sections_for_branch(branch)
        
        chainage = []
        minmax = []
        for cross_section in css:
            chainage.append(cross_section[1])
            roughness = self.getRoughnessInfoForCss(cross_section[0])[1]
            minmax.append([min(roughness), max(roughness)])
        
        chainage = np.array(chainage) * 1e-3
        minmax = np.array(minmax)
        ax.plot(chainage, minmax[:,0], label='minimum')
        ax.plot(chainage, minmax[:,1], label='maximum')
        ax.set_ylabel('Ruwheid (Chezy)')
        ax.set_xlabel('Afstand [km]')
        ax.set_title(branch)
        fig, lgd = self._SetPlotStyle(fig, use_legend=True)
        plt.savefig(self.fig_dir.joinpath(f'roughness_longitudinal_{branch}.png'), bbox_extra_artists=[lgd], bbox_inches='tight')
    
    
    def get_cross_sections_for_branch(self, branch:str):
        def split_css(name)->Tuple[str,float,str]:
            chainage = float(name.split("_")[-1])
            branch = "_".join(name.split("_")[:-1])
            return  (name, 
                        chainage,
                        branch)

        def get_css_for_branch(css_list, branchname:str):
            return [c for c in css_list if c[2].startswith(branchname)]

        css_list = [split_css(css.get('id')) for css in self.cross_sections]
        branches, contiguous_branches = self.branches
        branch_list = []
        sub_branches = np.unique([b for b in branches if b.startswith(branch)])
        running_chainage = 0
        for i, sub_branch in enumerate(sub_branches):
            sublist = get_css_for_branch(css_list, sub_branch)
            if i > 0: 
                running_chainage += get_css_for_branch(css_list, sub_branches[i-1])[-1][1]
            branch_list.extend([(s[0], s[1]+running_chainage, s[2]) for s in sublist])

        return branch_list 

    @property
    def branches(self)->Generator[List[str], None, None]:
        def split_css(name)->Tuple[str,float,str]:
            chainage = float(name.split("_")[-1])
            branch = "_".join(name.split("_")[:-1])
            return  (name, 
                        chainage,
                        branch)
        
        def find_branches(css_list)->List[str]:
            branches = np.unique([i[2] for i in css_names])
            contiguous_branches = np.unique([b.split("_")[0] for b in branches])
            return branches, contiguous_branches

        css_names = [split_css(css.get('id')) for css in self.cross_sections]
        branches, contiguous_branches  = find_branches(css_names)
        return branches, contiguous_branches
        

        


    def _generate_output_dir(
        self, figure_type: str = "png", overwrite: bool = True
    ):
        """
        Creates a new directory in the output map to store figures for each cross-section

        Arguments:
            output_map - path to fm2prof output directory

        Returns:
            png images saved to file
        """

        figdir = self.output_dir.joinpath('figures')
        if not figdir.is_dir(): figdir.mkdir(parents=True)
        return figdir

    def _set_files(self):
        self.files = {
            "css_def": os.path.join(self.output_dir, self.__cssdeffile),
            "volumes": os.path.join(self.output_dir, self.__volumefile),
            "roughnessMain": os.path.join(self.output_dir, self.__rmainfile),
            "roughnessFP1": os.path.join(self.output_dir, self.__rfp1file),
        }

    def _getValueFromLine(self, f):
        return f.readline().strip().split("=")[1].strip()

    def _readCSSDefFile(self)->List[Dict]:
        csslist = list()

        with open(self.files.get("css_def"), "r") as f:
            for line in f:
                if line.lower().strip() == "[definition]":
                    css_id = f.readline().strip().split("=")[1]
                    [f.readline() for i in range(3)]
                    css_levels = list(map(float, self._getValueFromLine(f).split()))
                    css_fwidth = list(map(float, self._getValueFromLine(f).split()))
                    css_twidth = list(map(float, self._getValueFromLine(f).split()))
                    css_sdcrest = float(self._getValueFromLine(f))
                    css_sdflow = float(self._getValueFromLine(f))
                    css_sdtotal = float(self._getValueFromLine(f))
                    css_sdbaselevel = float(self._getValueFromLine(f))
                    css_mainsectionwidth = float(self._getValueFromLine(f))
                    css_fp1sectionwidth = float(self._getValueFromLine(f))

                    css = {
                        "id": css_id.strip(),
                        "levels": css_levels,
                        "flow_width": css_fwidth,
                        "total_width": css_twidth,
                        "SD_crest": css_sdcrest,
                        "SD_flow_area": css_sdflow,
                        "SD_total_area": css_sdtotal,
                        "SD_baselevel": css_sdbaselevel,
                        "mainsectionwidth": css_mainsectionwidth,
                        "fp1sectionwidth": css_fp1sectionwidth,
                    }
                    csslist.append(css)

        return csslist

    def getRoughnessInfoForCss(self, cssname, rtype: str = "roughnessMain"):
        """
        Opens roughness file and reads information for a given cross-section
        name
        """
        levels = None
        values = None
        with open(self.files[rtype], "r") as f:
            cssbranch, csschainage = self._parse_cssname(cssname)
            for line in f:
                if line.strip().lower() == "[branchproperties]":
                    if self._getValueFromLine(f).lower() == cssbranch:
                        [f.readline() for i in range(3)]
                        levels = list(map(float, self._getValueFromLine(f).split()))
                if line.strip().lower() == "[definition]":
                    if self._getValueFromLine(f).lower() == cssbranch:
                        if float(self._getValueFromLine(f).lower()) == csschainage:
                            values = list(map(float, self._getValueFromLine(f).split()))
        return levels, values

    def getVolumeInfoForCss(self, cssname):
        column_names = [
            "z",
            "2D_total_volume",
            "2D_flow_volume",
            "2D_wet_area",
            "2D_flow_area",
            "1D_total_volume_sd",
            "1D_total_volume",
            "1D_flow_volume_sd",
            "1D_flow_volume",
            "1D_total_width",
            "1D_flow_width",
        ]
        cssdata = {}
        for column in column_names:
            cssdata[column] = list()

        with open(self.files["volumes"], "r") as f:
            for line in f:
                values = line.strip().split(",")
                if values[0] == cssname:
                    for i, column in enumerate(column_names):
                        cssdata[column].append(float(values[i + 1]))

        return cssdata

    @property
    def cross_sections(self) -> Generator[Dict,None,None]:
        """
        Generator to loop through all cross-sections in definition file.

        Example use:
        for css in visualiser.cross_sections():
            visualiser.make_figure(css)
        """
        csslist = self._readCSSDefFile()
        for css in csslist:
            yield css

    def get_cross_section_by_id(self, id: str) -> dict:
        """
        Get cross-section information given an id.

        Arguments:
            id (str): cross-section name
        """
        csslist = self._readCSSDefFile()
        for css in csslist:
            if css.get("id") == id:
                return css

    def figure_cross_section(
        self,
        css,
        reference_geometry: tuple = (),
        reference_roughness: tuple = (),
        save_to_file: bool = True,
    ):
        """
        Creates a figure

        Arguments

            css: dictionary containing cross-section information. Obtain with `VisualiseOutput.cross_sections`
                 generator or `VisualiseOutput.get_cross_section_by_id` method.

            reference_geometry (tuple): tuple(list(y), list(z))

            reference_roughness (tuple): tuple(list(z), list(n))

            save_to_file (bool): if true, save figure to VisualiseOutput.fig_dir
                                 if false, returns pyplot figure object

        """
        try:
            fig = plt.figure(figsize=(8, 12))
            gs = fig.add_gridspec(2, 2)
            axs = [
                fig.add_subplot(gs[0, :]),
                fig.add_subplot(gs[1, 0]),
                fig.add_subplot(gs[1, 1]),
            ]

            self._plot_geometry(css, axs[0], reference_geometry)
            self._plot_volume(css, axs[1])
            self._plot_roughness(css, axs[2], reference_roughness)

            fig, lgd = self._SetPlotStyle(fig)

            if save_to_file:
                print(f"saved to {self.fig_dir}/{css['id']}.png")
                plt.savefig(f"{self.fig_dir}/{css['id']}.png", bbox_extra_artists=[lgd], bbox_inches='tight')
            else:
                return fig

        except Exception as e:
            print(f"error processing: {css['id']} {str(e)}")
            return None

        finally:
            plt.close()

    def _SetPlotStyle(self, *args, **kwargs):
        """ todo: add preference to switch styles or
        inject own style
        """ 
        return PlotStyles.van_veen(*args, **kwargs)

    def _plot_geometry(self, css, ax, reference_geometry=None):

        # Get data
        tw = np.append([0], np.array(css["total_width"]))
        fw = np.append([0], np.array(css["flow_width"]))
        l = np.append(css["levels"][0], np.array(css["levels"]))
        mainsectionwidth = css["mainsectionwidth"]
        fp1sectionwidth = css["fp1sectionwidth"]

        # Get the water level where water level independent computation takes over
        # this is the lowest level where there is 2D information on volumes
        z_waterlevel_independent = self._get_lowest_water_level_in_2D(css)

        # Plot cross-section geometry
        for side in [-1, 1]:
            h = ax.fill_betweenx(
                l, side * fw / 2, side * tw / 2, color="#44B1D5AA", hatch="////"
            )
            ax.plot(side * tw / 2, l, "-k")
            ax.plot(side * fw / 2, l, "--k")

        # Plot roughness section width
        ax.plot(
            [-0.5 * mainsectionwidth, 0.5 * mainsectionwidth],
            [min(l) - 0.25] * 2,
            "-",
            linewidth=2,
            color="red",
            label="Main section",
        )
        ax.plot(
            [
                -0.5 * (mainsectionwidth + fp1sectionwidth),
                0.5 * (mainsectionwidth + fp1sectionwidth),
            ],
            [min(l) - 0.25] * 2,
            "--",
            color="red",
            label="Floodplain section",
        )
        # ax.plot(tw-0.5*max(fp1sectionwidth),[min(l)]*len(l), '--', color='cyan', label='Floodplain section')

        # Plot water level indepentent line
        ax.plot(
            tw - 0.5 * max(tw),
            [z_waterlevel_independent] * len(l),
            linestyle="--",
            color="m",
            label="Lowest water level in 2D",
        )

        h.set_label("Storage")

        ax.set_title(css["id"])
        ax.set_xlabel("[m]")
        ax.set_ylabel("[m]")

        if reference_geometry:
            ax.plot(
                reference_geometry[1],
                reference_geometry[0],
                "--r",
                label="Reference geometry",
            )

        # Plot crest level height
        sd_info = self._get_sd_plot_info(css)

        ax.plot(
            [-tw[-1] / 2, tw[-1] / 2],
            [sd_info.get("crest")] * 2,
            linestyle=sd_info.get("linestyle"),
            linewidth=1,
            color="orange",
            label=sd_info.get("label"),
        )

    def _plot_volume(self, css, ax):
        # Get data
        vd = self.getVolumeInfoForCss(css["id"])
        z_waterlevel_independent = self._get_lowest_water_level_in_2D(css)

        # Plot 1D volumes
        ax.fill_between(
            vd["z"],
            0,
            vd["1D_total_volume_sd"],
            color="#24A493",
            label="1D Total Volume (incl. SD)",
        )
        ax.fill_between(
            vd["z"],
            0,
            vd["1D_total_volume"],
            color="#108A7A",
            label="1D Total Volume (excl. SD)",
        )
        ax.fill_between(
            vd["z"],
            0,
            vd["1D_flow_volume"],
            color="#209AB4",
            label="1D Flow Volume (incl. SD)",
        )
        ax.fill_between(
            vd["z"],
            0,
            vd["1D_flow_volume"],
            color="#0C6B7F",
            label="1D Flow Volume (excl. SD)",
        )
        # Plot 2D volume
        ax.plot(vd["z"], vd["2D_total_volume"], "--k", label="2D Total Volume")
        ax.plot(vd["z"], vd["2D_flow_volume"], "-.k", label="2D Flow Volume")

        # Plot lowest point in 2D
        ax.plot(
            [z_waterlevel_independent] * len(vd["2D_total_volume"]),
            vd["2D_total_volume"],
            linestyle="--",
            linewidth=1,
            color="m",
            label="Lowest water level in 2D",
        )

        # Plot SD crest
        sd_info = self._get_sd_plot_info(css)
        ax.plot(
            [sd_info.get("crest")] * len(vd["2D_total_volume"]),
            vd["2D_total_volume"],
            linestyle=sd_info.get("linestyle"),
            linewidth=1,
            color="orange",
            label=sd_info.get("label"),
        )

        ax.set_ylim([0, ax.get_ylim()[1]])
        ax.set_xlim([min(vd["z"]), max(vd["z"])])
        ax.set_title("Volume graph")
        ax.set_xlabel("Water level [m]")
        ax.set_ylabel("Volume [m$^3$]")

    def _plot_roughness(self, css, ax, reference_roughness):
        levels, values = self.getRoughnessInfoForCss(css["id"], rtype="roughnessMain")
        try:
            ax.plot(levels, values, label="Main channel")
        except:
            pass

        try:
            levels, values = self.getRoughnessInfoForCss(
                css["id"], rtype="roughnessFP1"
            )
            if levels is not None:
                ax.plot(levels, values, label="Floodplain1")
        except FileNotFoundError:
            pass

        if reference_roughness:
            ax.plot(
                reference_roughness[0],
                reference_roughness[1],
                "--r",
                label="Reference friction",
            )

        # Limit x axis to min and maximum level in cross-section
        ax.set_xlim(min(css["levels"]), max(css["levels"]))
        ax.set_title("Roughness")
        ax.set_xlabel("Water level [m]")
        ax.set_ylabel("Chezy coefficient [m$^{1/2}$/s]")

    @staticmethod
    def _get_sd_plot_info(css):
        l = np.append(css["levels"][0], np.array(css["levels"]))
        z_crest_level = css["SD_crest"]
        if z_crest_level <= max(l):
            if z_crest_level >= min(l):
                sd_linestyle = "--"
                sd_label = "SD Crest Level"
            else:
                z_crest_level = min(l)
                sd_linestyle = "-"
                sd_label = "SD Crest Level (cropped)"
        else:
            z_crest_level = max(l)
            sd_linestyle = "-"
            sd_label = "SD Crest Level (cropped)"

        return {"linestyle": sd_linestyle, "label": sd_label, "crest": z_crest_level}

    def _get_lowest_water_level_in_2D(self, css):
        vd = self.getVolumeInfoForCss(css["id"])
        index_waterlevel_independent = np.argmax(~np.isnan(vd.get("2D_total_volume")))
        z_waterlevel_independent = vd.get("z")[index_waterlevel_independent]
        return z_waterlevel_independent

    def _parse_cssname(self, cssname):
        """
        returns name of branch and chainage
        """
        branch, chainage = cssname.rsplit(
            "_", 1
        )  # rsplit prevents error if branchname contains _
        chainage = round(float(chainage), 2)

        return branch, chainage


class PlotStyles:
    myFmt = mdates.DateFormatter('%d-%b')
    monthlocator = mdates.MonthLocator() 
    daylocator = mdates.DayLocator(15)

    @staticmethod
    def _is_timeaxis(axis)->bool:
        try:
            float(axis.get_ticklabels()[0].get_text().replace('−', '-'))
        except ValueError:
            return True
        return False

    @classmethod
    def pilot_2021(cls, fig, legendbelow=False):
        for ax in fig.axes:
            ax.grid(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            ax.xaxis.set_tick_params(width=2)
            ax.yaxis.set_tick_params(width=2)
            for spine in ["left", "bottom"]:
                ax.spines[spine].set_edgecolor("#272727")
                ax.spines[spine].set_linewidth(2)
            if legendbelow:
                legend = ax.legend(
                    fancybox=True,
                    framealpha=0.5,
                    edgecolor="None",
                    loc=3,
                    ncol=3,
                    bbox_to_anchor=(-0.02, -0.5),
                )
            else:
                legend = ax.legend(fancybox=True, framealpha=0.5, edgecolor="None")
            legend.get_frame().set_facecolor("#e5eef2")  # #e5eef2 #92b6c7
            legend.get_frame().set_boxstyle("square", pad=0)

    @classmethod
    def van_veen(cls, fig, use_legend:bool=True):
        """ Stijl van Van Veen """
        fig.canvas.draw()  # this forces labels to be generated
        font = {'family' : 'Bahnschrift',
                'weight' : 'normal',
                'size'   : 18}
        mpl.rc('font', **font)
        mpl.rcParams["axes.unicode_minus"] = False
        legend_title= r'toelichting'

        handles = list()
        labels = list()

        for ax in fig.axes:
            
            ax.grid(b=True, which="major", linestyle="-", linewidth=1, color='k')
            ax.grid(b=True, which="minor", linestyle="-", linewidth=0.5, color='k')
            
            for _, spine in ax.spines.items():
                spine.set_linewidth(2)
            
            
            if cls._is_timeaxis(ax.xaxis):  
                ax.xaxis.set_major_formatter(cls.myFmt)
                ax.xaxis.set_major_locator(cls.monthlocator)
                ax.xaxis.set_minor_locator(cls.daylocator)
            if cls._is_timeaxis(ax.yaxis):
                ax.yaxis.set_major_formatter(cls.myFmt)
                ax.yaxis.set_major_locator(cls.monthlocator)
                ax.yaxis.set_minor_locator(cls.daylocator)

            """
            if legend:
                ax.legend(loc='best',
                          edgecolor="k", 
                          facecolor='white',
                          framealpha=1,
                          borderaxespad=0,
                          title=legend_title.upper())
            
            """
            ax.set_title(ax.get_title().upper())
            ax.set_xlabel(ax.get_xlabel().upper())
            ax.set_ylabel(ax.get_ylabel().upper())

        
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)

        fig.tight_layout()
        if use_legend:
            lgd = fig.legend(handles, labels, 
                            loc= 'upper left',
                            bbox_to_anchor=(1.0, 0.9),
                            bbox_transform=fig.transFigure,
                            edgecolor="k", 
                            facecolor='white',
                            framealpha=1,
                            borderaxespad=0,
                            title=legend_title.upper())

            return fig, lgd
        else:
            return fig, handles, labels