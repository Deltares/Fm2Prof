"""
Copyright (C) Stichting Deltares 2019. All rights reserved.

This file is part of the Fm2Prof.

The Fm2Prof is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.

All names, logos, and references to "Deltares" are registered trademarks of
Stichting Deltares and remain full property of Stichting Deltares at all times.
All rights reserved.
"""

"""
Utilities for FM2PROF 
"""

import numpy as np 
import os
import shutil
import matplotlib.pyplot as plt

def SetDeltaresStyle(fig, legendbelow=False):
    for ax in fig.axes:
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.xaxis.set_tick_params(width=2)
        ax.yaxis.set_tick_params(width=2)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_edgecolor('#272727')
            ax.spines[spine].set_linewidth(2)
        if legendbelow:
            legend = ax.legend(fancybox=True, 
                               framealpha=0.5, 
                               edgecolor='None',
                               loc=3,
                               ncol=3,
                               bbox_to_anchor=(-0.02, -0.5))
        else:
            legend = ax.legend(fancybox=True, 
                               framealpha=0.5, 
                               edgecolor='None')
        legend.get_frame().set_facecolor('#e5eef2')  # #e5eef2 #92b6c7
        legend.get_frame().set_boxstyle("square", pad=0)
    

def networkdeffile_to_input(networkdefinitionfile, crossectionlocationfile):
    """
    Builds a cross-section input file for FM2PROF from a DIMR network definition file. 

    The distance between cross-section is computed from the differences between the offsets/chainages. 
    The beginning and end point of each branch are treated as half-distance control volumes. 
    """

    # Open network definition file, for each branch extract necessary info
    x = []          # x-coordinate of cross-section centre
    y = []			# y-coordinate of cross-section centre
    cid = []		# id of cross-section
    bid = []		# id of 1D branch
    coff = []		# offset of cross-section on 1D branch ('chainage')
    cdis = []		# distance of 1D branch influenced by crosss-section ('vaklengte')

    with open(networkdefinitionfile, 'r') as f:
        for line in f:
            if line.strip().lower() == "[branch]":
                branchid = f.readline().split('=')[1].strip()
                xlength = 0
                for i in range(10):
                    bline = f.readline().strip().lower().split('=')
                    if bline[0].strip() == "gridpointx":
                        xtmp = list(map(float, bline[1].split()))
                    elif bline[0].strip() == "gridpointy":
                        ytmp = list(map(float, bline[1].split()))
                    elif bline[0].strip() == "gridpointids":
                        cidtmp = bline[1].split(';')
                    elif bline[0].strip() == "gridpointoffsets":
                        cofftmp = list(map(float, bline[1].split()))

                        # compute distance between control volumes
                        cdistmp = np.append(np.diff(cofftmp)/2, [0]) + np.append([0], np.diff(cofftmp)/2)

                # Append branchids
                bid.extend([branchid] * len(xtmp))

                # Correct end points (: at end of branch, gridpoints of this branch and previous branch
                # occupy the same position, which does not go over well with fm2profs classification algo)
                offset = 1
                xtmp[0] = np.interp(offset, cofftmp, xtmp)
                ytmp[0] = np.interp(offset, cofftmp, ytmp)
                offset = cofftmp[-1]-1
                xtmp[-1] = np.interp(offset, cofftmp, xtmp)
                ytmp[-1] = np.interp(offset, cofftmp, ytmp)


                # Append all poitns
                x.extend(xtmp)
                y.extend(ytmp)
                cid.extend([c.strip() for c in cidtmp])
                coff.extend(cofftmp)
                cdis.extend(cdistmp)


    with open(crossectionlocationfile, 'w') as f:
        f.write('name,x,y,length,branch,offset\n')
        for i in range(len(x)):
            f.write(f'{cid[i]}, {x[i]:.4f}, {y[i]:.4f}, {cdis[i]:.2f}, {bid[i]}, {coff[i]:.2f}\n')
            #f.write('{}, {:.4f}, {:.4f}, {:.2f}, {}, {:.2f}\n'.format(cid[i], x[i], y[i], cdis[i], bid[i], coff[i]))

class VisualiseOutput():
    __cssdeffile = 'CrossSectionDefinitions.ini'
    __volumefile = 'volumes.csv'
    __rmainfile  = 'roughness-Main.ini'
    __rfp1file  = 'roughness-FloodPlain1.ini'

    def __init__(self, output_directory: str, figure_type: str="png", overwrite: bool=True):
        self.output_dir = output_directory
        self.fig_dir = self.generate_output_dir(output_directory)
        self._set_files()

    def generate_output_dir(self, output_directory, figure_type: str="png", overwrite: bool=True):
        """
        Creates a new directory in the output map to store figures for each cross-section

        Arguments:
            output_map - path to fm2prof output directory

        Returns:
            png images saved to file
        """
        

        figdir = os.path.join(output_directory, 'figures')
        if os.path.isdir(figdir) & overwrite:
            shutil.rmtree(figdir)
            os.mkdir(figdir)
        elif not os.path.isdir(figdir):
            os.mkdir(figdir)
        return figdir

    def _set_files(self):
        self.files = {'css_def': os.path.join(self.output_dir, self.__cssdeffile),
                      'volumes': os.path.join(self.output_dir, self.__volumefile),
                      'roughnessMain': os.path.join(self.output_dir, self.__rmainfile),
                      'roughnessFP1': os.path.join(self.output_dir, self.__rfp1file)
                      }

    def _getValueFromLine(self, f):
        return f.readline().strip().split('=')[1].strip()

    def _readCSSDefFile(self):
        csslist = list()

        with open(self.files.get('css_def'), 'r') as f:
            for line in f:
                if line.lower().strip() == "[definition]":
                    css_id = f.readline().strip().split('=')[1]
                    [f.readline() for i in range(3)]
                    css_levels = list(map(float, self._getValueFromLine(f).split()))
                    css_fwidth = list(map(float, self._getValueFromLine(f).split()))
                    css_twidth = list(map(float, self._getValueFromLine(f).split()))
                    css_sdcrest = float(self._getValueFromLine(f))
                    css_sdflow = float(self._getValueFromLine(f))
                    css_sdtotal = float(self._getValueFromLine(f))

                    css = {"id":css_id.strip(),
                           "levels": css_levels,
                           "flow_width": css_fwidth,
                           "total_width": css_twidth,
                           "SD_crest": css_sdcrest,
                           "SD_flow_area": css_sdflow,
                           "SD_total_area": css_sdtotal
                           }
                    csslist.append(css)

        return csslist

    def getRoughnessInfoForCss(self, cssname, rtype: str='roughnessMain'):
        levels = None
        values = None
        with open(self.files[rtype], 'r') as f:
            cssbranch, csschainage = cssname.split('_')
            for line in f:
                if line.strip().lower() == '[branchproperties]':
                    if self._getValueFromLine(f).lower()==cssbranch:
                        [f.readline() for i in range(3)]
                        levels = list(map(float, self._getValueFromLine(f).split(',')))
                if line.strip().lower() == '[definition]':
                    if self._getValueFromLine(f).lower()==cssbranch:
                        if float(self._getValueFromLine(f).lower()) == float(csschainage):
                            values = list(map(float, self._getValueFromLine(f).split(',')))
        return levels, values

    def getVolumeInfoForCss(self, cssname):
        column_names = ["z","2D_total_volume","2D_flow_volume","2D_wet_area","2D_flow_area","1D_total_volume_sd",
                        "1D_total_volume","1D_flow_volume_sd","1D_flow_volume","1D_total_width","1D_flow_width"]
        cssdata = {}
        for column in column_names:
            cssdata[column] = list()

        with open(self.files['volumes'], 'r') as f:
            for line in f:
                values = line.strip().split(',')
                if values[0] == cssname:
                    for i, column in enumerate(column_names):
                        cssdata[column].append(float(values[i+1]))

        return cssdata      
                    
    def save_to_file(self):
        csslist = self._readCSSDefFile()
        for css in csslist:
            try:
                fig = plt.figure(figsize=(8, 12))
                gs = fig.add_gridspec(2, 2)
                axs = [fig.add_subplot(gs[0, :]),
                       fig.add_subplot(gs[1,0]),
                       fig.add_subplot(gs[1,1])]
                tw = np.append([0], np.array(css['total_width']))
                fw = np.append([0], np.array(css['flow_width']))
                l = np.append(css['levels'][0], np.array(css['levels']))

                # Plot cross-section geometry
                for side in [-1, 1]:
                    axs[0].fill_betweenx(l, side*fw/2, side*tw/2, color="#44B1D5AA", hatch='////', label='Storage ')
                    axs[0].plot(side*tw/2, l, '-k')
                    axs[0].plot(side*fw/2, l, '--k')
                axs[0].set_title(css['id'])
                axs[0].set_xlabel('[m]')
                axs[0].set_ylabel('[m]')

                # Plot Volume
                vd = self.getVolumeInfoForCss(css["id"])

                axs[1].fill_between(vd['z'], 0, vd['1D_total_volume_sd'], 
                                    color="#24A493",
                                    label='1D Total Volume (incl. SD)')
                axs[1].fill_between(vd['z'], 0, vd['1D_total_volume'], 
                                    color="#108A7A",
                                    label='1D Total Volume (excl. SD)')
                axs[1].fill_between(vd['z'], 0, vd['1D_flow_volume'], 
                                    color="#209AB4",
                                    label='1D Flow Volume (incl. SD)')
                axs[1].fill_between(vd['z'], 0, vd['1D_flow_volume'], 
                                    color="#0C6B7F",
                                    label='1D Flow Volume (excl. SD)')

                axs[1].plot(vd['z'], vd['2D_total_volume'], '--k', label='2D Total Volume')
                axs[1].plot(vd['z'], vd['2D_flow_volume'], '-.k', label='2D Flow Volume')

                
                axs[1].set_title('Volume graph')
                axs[1].set_xlabel('Water level [m]')
                axs[1].set_ylabel('Volume [m$^3$]')

                # Plot Roughness
                levels, values = self.getRoughnessInfoForCss(css["id"], rtype='roughnessMain')
                axs[2].plot(levels, values,  label='Main channel')

                levels, values = self.getRoughnessInfoForCss(css["id"], rtype='roughnessFP1')
                axs[2].plot(levels, values, label='Floodplain1')

                axs[2].set_xlabel('Water level [m]')
                axs[2].set_ylabel('Manning coefficient [sm$^{-1/3}$]')

                SetDeltaresStyle(fig)
                plt.tight_layout()
                plt.savefig(f"{self.fig_dir}/{css['id']}.png")
                print (f"processed {css['id']}")
            except Exception:
                print (f"error processing: {css['id']}")
            finally:
                plt.close()

