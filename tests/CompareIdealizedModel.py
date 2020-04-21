import os
import shutil
import numpy as np
import pandas as pd
from typing import Tuple, List
from tests.TestUtils import TestUtils

import matplotlib
import matplotlib.pyplot as plt


class CompareHelper:
    
    @staticmethod
    def interpolate_to_css(css: dict, xyz: Tuple[List[float]]):
        """
        interpolate the analytical cross-section
        at the given chainage y (float)
        """
        chainage = float(css.get('id').split('_')[1])
        xyz = np.array(xyz)

        # get cross-section closest to chainage
        chainages = xyz[:, 0]
        closest_chainage = chainages[np.argmin(abs(chainages - chainage))]
        css_template = xyz[xyz[:, 0] == closest_chainage]

        # Shift downward 
        z_out = []
        for y in np.unique(xyz[:, 1]):
            mask = xyz[:, 1] == y
            z_out.append(np.interp(chainage, xyz[mask, 0], xyz[mask, -1]))
        
        w_out = np.interp(css['levels'], z_out, css_template[:, 1])

        return css['levels'], w_out

    @staticmethod
    def convert_ZW_to_symmetric_css(gmlist):
        gmlist = np.array([np.array(xi) for xi in gmlist])
        return [np.append(list(reversed(gmlist[0])), gmlist[0]), 
                np.append(list(reversed(-gmlist[1]/2)), gmlist[1]/2)]

    @staticmethod
    def get_analytical_roughness_for_case(
            self, H_pos, b: float, y, MainChannel, case_name):

        
    
class CompareIdealizedModel:
    def _compare_volume(
            self, case_name, input_volume_file, fm2prof_fig_dir):
        # Read data
        df = pd.read_csv(input_volume_file, index_col=1)

        # Loop over each cross-sections
        crosssections = np.unique(df.id)
        
        for crosssection in crosssections:
            # Plot volume at each cross-section
            self.__plot_volume(df, crosssection, fm2prof_fig_dir)

    def _compare_css(
            self, case_name, tzw_values, input_geometry_file, fm2prof_fig_dir):

        #  Read data
        (Z, W, F, Y, CL, FPB, FA, TA) = self.__get_geometry_data(
            input_geometry_file)

        #  Check whether flow width exists
        S = self.__FlowWidth_check(W, F, case_name)

        #  Loop over each chainage (cross-section)
        for cs in range(len(Y)):
            y = Y[cs]
            z = Z[cs]
            w = W[cs]
            s = S[cs]
            # Get the interpolated analytical result at chainage y
            [tz, tx] = self.__interpolate_z(tzw_values, y)

            # check whether the analytical cross-section is symmetric
            if not self.__symmetric(tx):
                tx = self.__ShiftCrossSection(tx, tz)
            [plt_z, plt_x] = self.__convert_zw2xz(z, w, tx, max(w))
            [plt_tz, plt_tx] = self.__Interpolate_tz_for_CS(
                plt_z, plt_x, tz, tx)

            plt_s_flag = False
            if sum(s) > 0.001 or case_name == 'case_04_storage':
                # Storage or false storage width
                plt_s = None
                ts = None
                plt_s_flag = True
                # case04_storage case only
                if Y[cs] >= 1250 and Y[cs] <= 1750 and case_name == 'case_04_storage':
                    ts = self.__interpolate_s()
                if sum(s) > 0.001:
                    [plt_z, plt_s] = self.__convert_zw2xz(z, s, tx, max(w), 1)
                    plt_s = [plt_s[i]+plt_x[i] for i in range(len(plt_s))]
            if case_name == 'case_05_dyke':
                # Dike case
                # 50m * 1m (crest level-base level) * 2
                # = analytical total area behind summer dike
                ttbs = 2 * 50.0 * 1.0
                # real floodplain is 1m lower than the "floodplain" in the
                # cross-section geometry
                tfpb = tz[1] - 1.0
                # real crest level is the "floodplain"
                # height in the cross-section geometry
                tcl = tz[1]
                # crest level from fm2prof
                cl = CL[cs]
                # floodplain base level from fm2prof
                fpb = FPB[cs]
                # total area behind summer dike from fm2prof
                tbs = TA[cs]
            #  Derive the error values
            sumError = self.__Error_check(plt_tz, plt_z, plt_x)

            # Plot the result
            if plt_s_flag:
                if case_name == 'case_05_dyke':
                    self.__plot_cs(
                        fm2prof_fig_dir, tx, tz, plt_x, plt_z, y,
                        sumError, plt_s, ts, ttbs, tfpb, tcl, tbs, fpb, cl)
                else:
                    self.__plot_cs(
                        fm2prof_fig_dir, tx, tz, plt_x, plt_z, y,
                        sumError, plt_s, ts)
            else:
                if case_name == 'case_05_dyke':
                    self.__plot_cs(
                        fm2prof_fig_dir, tx, tz, plt_x, plt_z, y,
                        sumError, '', '', ttbs, tfpb, tcl, tbs, fpb, cl)
                else:
                    self.__plot_cs(
                        fm2prof_fig_dir, tx, tz, plt_x, plt_z, y,
                        sumError)

    def _compare_roughness(
            self, case_name, tzw_values, input_roughness_file, fm2prof_fig_dir):
        # Read data in roughness.csv
        (S, Y, N, H_pos, R_pos) = self.__get_roughness_data(input_roughness_file)

        # loop over each chainage (cross-section)
        y0 = 0
        MainChannel = True      # Main channel
        for cs in range(len(Y)):
            if S[cs].lower() != "main":
                MainChannel = False
            else:
                MainChannel = True
            y = Y[cs]           # chainage
            # n = n0 + N[cs]      # maximum index
            hpos = H_pos[cs]  # H_pos at chainage y
            rpos = R_pos[cs]  # R_pos at chainage y
            # n0 = n              # set up n0 at the next chainage
            # give the interpolated analytical result at chainage y
            [tz, tx] = self.__interpolate_z(tzw_values, y)
            # get the bottom level to determine the water depth
            baselevel = min(tz)
            # analytical roughness
            (tH_pos, tR_pos) = self.__get_analytical_roughness(
                    hpos, baselevel, y, MainChannel, case_name)
            y0 = y

            # plot the fm2prof and analytical roughness (H_pos vs Cz)
            self.__plot_roughness(
                    fm2prof_fig_dir, tH_pos, tR_pos,
                    hpos, rpos, y, MainChannel)

    def __plot_volume(self, df, cs, output_folder):
        _, ax = plt.subplots(1, figsize=(10, 4))
        tv_1d = df[df.id==cs]['1D_total_volume']
        tv_1dsd = df[df.id==cs]['1D_total_volume_sd']
        fv_1d = df[df.id==cs]['1D_flow_volume_sd']
        tv_2d = df[df.id==cs]['2D_total_volume']
        fv_2d = df[df.id==cs]['2D_flow_volume']
        ax.plot(tv_2d, '-', linewidth=5, color=[0.4]*3, label='2D Total volume')
        ax.plot(fv_2d, '--', linewidth=4, color=[0.8, 0, 0.8], label='2D Flow volume')	
        ax.plot(tv_1dsd, '--r', label='1D Total volume (+sd)')
        ax.plot(fv_1d, '-.c', label='1D Flow volume (+sd)')
        ax.set_title(cs)
        ax.legend()
        ax.set_xlabel('Water level [m]')
        ax.set_ylabel('Volume [m$^3$]')
        plt.grid()
        cs_no_period = cs.replace('.','x') # latex won't allow periods 
        plt.savefig(os.path.join(output_folder, "{}_volumegraph.png".format(cs_no_period)))

    def __get_geometry_data(self, input_file: str):
        """[summary]

        Arguments:
            input_file {str} -- Geometry csv file

        Returns:
            {tuple} --  Y = chainage; CL = crest level;
                        FPB = floodplain base level;
                        FA = flow area behind summer dike;
                        TA = total area behind summer dike
                        Z = cross-section z values;
                        W = cross-section total width;
                        F = cross-section flow width
        """
        # Reading geometry.csv file
        # Output lists ===

        z_tmp = []
        w_tmp = []
        f_tmp = []
        Y = []
        CL = []
        FPB = []
        FA = []
        TA = []
        n = 0
        assert os.path.exists(input_file), '' + \
            'Input file {} does not exist'.format(input_file)

        with open(input_file) as fin:
            for line in fin:
                ls = line.strip().split(',')
                if 'id' in line[:2]:
                    z_index = ls.index('level')
                    w_index = ls.index('Total width')
                    f_index = ls.index('Flow width')
                    y_index = ls.index('chainage')
                    cl_index = ls.index('Crest level summerdike')
                    fpb_index = ls.index(
                        'Floodplain baselevel behind summerdike')
                    fa_index = ls.index('Flow area behind summerdike')
                    ta_index = ls.index('Total area behind summerdike')
                    sd_key = ls.index('Use Summerdike')
                elif 'meta' ''in line:
                    Y.append(float(ls[y_index]))      # chainage
                    if ls[sd_key] == '1':
                        # crest level
                        CL.append(float(ls[cl_index]))
                        # floodplain base level
                        FPB.append(float(ls[fpb_index]))
                        # flow area behind summer dike
                        FA.append(float(ls[fa_index]))
                        # total area behind summer dike
                        TA.append(float(ls[ta_index]))
                    if n == 1:
                        Z = [z_tmp]
                        W = [w_tmp]
                        F = [f_tmp]
                        n += 1
                    elif n > 1:
                        Z.append(z_tmp)
                        W.append(w_tmp)
                        F.append(f_tmp)
                    z_tmp = []
                    w_tmp = []
                    f_tmp = []
                    n += 1
                elif 'geom' in line:
                    z_tmp.append(float(ls[z_index]))  # z values
                    w_tmp.append(float(ls[w_index]))  # w values (total width)
                    f_tmp.append(float(ls[f_index]))  # fw values (flow width)
            Z.append(z_tmp)
            W.append(w_tmp)
            F.append(f_tmp)

        return (Z, W, F, Y, CL, FPB, FA, TA)

    def __interpolate_z(self, tzw: list, y: float):
        """
        interpolate the analytical cross-section
        at the given chainage y (float)
        """
        tz = []
        tw = []
        y0 = tzw[0][1]
        for i in range(len(tzw)):
            if tzw[i][0] == y:
                tz.append(float(tzw[i][-1]))  # z
                tw.append(float(tzw[i][1]))  # y

            elif tzw[i][0] > y:
                for j in range(i-1, -1, -1):
                    if tzw[j][1] == y0:
                        tzw0 = j
                        tzw1 = i
                        ty0 = tzw[j][0]
                        tz0 = [tzw[z][-1] for z in range(tzw0, tzw1)]
                        tw0 = [tzw[z][1] for z in range(tzw0, tzw1)]
                        ty1 = tzw[i][0]
                        tz1_range = range(tzw1, tzw1 + tzw1 - tzw0)
                        tz1 = [tzw[z][-1] for z in tz1_range]
                        rr = (y-ty0)/(ty1-ty0)
                        tz1_len = range(len(tz1))
                        tz = [(rr*(tz1[x]-tz0[x]))+tz0[x] for x in tz1_len]
                        tw = tw0
                        break
                break
        return tz, tw

    def __Interpolate_tz_for_CS(self, plt_z, plt_x, tz, tx):
        est_z = [plt_z[0]]
        est_x = plt_x
        n = 0
        for i in range(1, len(plt_z) - 1):
            flag1 = 0
            for j in range(n, len(tz)):
                # est_x[i] == float(tx[j]):
                if abs(est_x[i]-float(tx[j])) < 1e-5:
                    est_z.append(tz[j])
                    n = j+1
                    break
                elif est_x[i] < float(tx[j]) and est_x[i] > float(tx[j-1]):
                    est_z.append(tz[j])
                    break
                elif est_x[i] > tx[-1]:
                    est_z.append(plt_z[i])
                    break
                elif abs(est_x[i]-est_x[i+1]) < 1e-5 and \
                        abs(est_x[i] - est_x[i-1]) < 1e-5:
                    est_z.append(tz[-1])
                    flag1 = 1
                    break
            if abs(est_x[i]-est_x[i+1]) < 1e-5 and \
                    abs(est_x[i]-est_x[i-1]) < 1e-5 and flag1 == 0:
                est_z.append(tz[-1])

        est_z.append(plt_z[-1])
        return est_z, est_x

    def __Error_check(self, plt_tz, plt_z, plt_x):
        diff = [plt_tz[i]-plt_z[i] for i in range(len(plt_z))]
        ErrorList = []
        for i in range(len(plt_tz)-1):
            dx = plt_x[i+1]-plt_x[i]
            diff_pos_i = diff[i]
            diff_next_pos = diff[i+1]
            if (diff_pos_i > 0 and diff_next_pos > 0) or \
                    (diff_pos_i < 0 and diff_next_pos < 0):
                min_z = abs(diff_pos_i)
                if abs(diff_pos_i) > abs(diff_next_pos):
                    min_z = abs(diff_next_pos)
                dx = plt_x[i+1]-plt_x[i]
                Error_area = (min_z + 0.5*abs(diff_next_pos-diff_pos_i)) * dx
            elif diff_pos_i == 0 and diff_next_pos == 0:
                Error_area = 0
            elif diff_pos_i == 0 or diff_next_pos == 0:
                dz = abs(diff_pos_i)
                if abs(diff_pos_i) < abs(diff_next_pos):
                    dz = abs(diff_next_pos)
                Error_area = 0.5 * dz * dx
            else:
                D = abs(diff_pos_i/diff_next_pos)
                dx0 = (D*dx)/(D+1)
                dx1 = dx-dx0
                diff_pos_i_dx0 = abs(diff_pos_i) * dx0
                diff_next_pos_dx1 = abs(diff_next_pos) * dx1
                Error_area = 0.5 * (diff_pos_i_dx0 + diff_next_pos_dx1)
            ErrorList.append(Error_area)
        return sum(ErrorList)

    def __ShiftCrossSection(self, tx, tz):
        min_value = min(tz)
        min_list = [i for i, x in enumerate(tz) if x == min_value]
        midpoint = (tx[min_list[-1]]-tx[min_list[0]])/2 + tx[min_list[0]]
        cs_midpoint = (tx[-1]-tx[0])/2 + tx[0]
        shift = midpoint - cs_midpoint
        tmp_tx = [tx[i]-shift for i in range(1, len(tx)-1)]
        new_tx = [tx[0]] + tmp_tx + [tx[-1]]
        return new_tx

    def __symmetric(self, L):
        if len(L) % 2 != 0:
            return False
        else:
            for i in range(1, int(len(L)/2)):
                if L[i]-L[i-1] != L[-i]-L[-i-1] > 1e-4:
                    return False
        return True

    def __plot_cs(
            self, fig_dir: str,
            tx, tz, plt_x, plt_z, y, err,
            plt_s=None, ts=None, ttbs=None, tfpb=None,
            tcl=None, tbs=None, fpb=None, cl=None):
        fig, axh = plt.subplots(1, figsize=(10, 4))
        tz_plt = [plt_z[0]] + tz + [plt_z[-1]]
        tx_plt = [tx[0]] + tx + [tx[-1]]
        if ts is not None and ts:
            ts_plt = [ts[0]] + ts + [ts[-1]]
            axh.plot(
                ts_plt, tz_plt,
                label='Analytical flow width',
                linestyle='--',
                color='#1f77b4')
        if plt_s is not None and plt_s:
            axh.plot(
                plt_s, plt_z,
                label='FM2PROF flow width',
                linestyle=':',
                color='#ff7f0e')
        axh.plot(
            tx_plt, tz_plt,
            label='Analytical total width',
            color='#ff7f0e')
        axh.plot(
            plt_x, plt_z,
            label='FM2PROF total width',
            color='#1f77b4')
        axh.set_ylim()
        axh.set_xlabel('x [m]')
        axh.set_ylabel('z [m]')
        axh.legend()
        titlestr = 'Cross-section at chainage ' + str(y)
        axh.set_title(titlestr)
        axh.text(
            0.06, 0.85, 'sum(err) = {:.2f}'.format(err),
            horizontalalignment='left',
            verticalalignment='center',
            transform=axh.transAxes)
        if ttbs is not None:
            ttbs_text = '' + \
                '\tCrestLevel = {:.2f}m,' + \
                ' Floodplain Base level = {:.2f}m\n' + \
                '\tCrest height = {:.2f}m,' + \
                ' Total area behind summer dike = {:.2f}m$^2$'
            axh.text(
                0.06, 0.72,
                'FM2PROF:\n' + ttbs_text.format(cl, fpb, cl - fpb, tbs),
                horizontalalignment='left',
                verticalalignment='center',
                transform=axh.transAxes)
            axh.text(
                0.06, 0.52,
                'Analytical:\n' + ttbs_text.format(tcl, tfpb, 1, ttbs),
                horizontalalignment='left',
                verticalalignment='center',
                transform=axh.transAxes)
        plt.grid()
        plt.tight_layout()
        figtitlestr = 'CrossSection_chainage' + str(int(y))
        fig_name = '{}.png'.format(figtitlestr)
        fig_path = os.path.join(fig_dir, fig_name)
        plt.savefig(fig_path)

    def __convert_zw2xz(
            self,
            z: list, w: list, tx: list, max_width: float,
            flow_width=None):
        """ Convert zw lists to xz list for plotting """
        if flow_width:
            w_tmp = [w[i]/2 for i in range(len(w))]
            plt_w_forward = [w_tmp[i]*-1 for i in range(len(w_tmp))]
            plt_w_reverse = w_tmp[::-1]
        else:
            w_tmp = [(max_width-w[i])/2 for i in range(len(w))]
            plt_w_forward = [max_width/2+w[i]/2 for i in range(len(w))]
            plt_w_reverse = w_tmp[::-1]
        plt_w_tmp = plt_w_reverse + plt_w_forward
        plt_w = [plt_w_tmp[i] + tx[0] for i in range(len(plt_w_tmp))]
        plt_z = z[::-1] + z
        return plt_z, plt_w

    def __FlowWidth_check(self, W: list, F: list, case_name: str):
        for i in range(len(W)):
            storage_width = [W[i][j]-F[i][j] for j in range(len(W[i]))]
            if i == 0:
                S = [storage_width]
            else:
                S.append(storage_width)
        return S

    def __interpolate_s(self):
        """ interpolate the analytical storage width """
        ts = [25, 50, 50.000001, 99.999999, 100, 125]
        return ts

    def __plot_roughness(
            self, fig_dir, tH_pos, tR_pos, H_pos,
            R_pos, y, MainChannel):
        fig, axh = plt.subplots(1, figsize=(10, 4))
        axh.plot(tH_pos, tR_pos, label='Analytical roughness',color='#ff7f0e')
        axh.plot(H_pos, R_pos, label='FM2PROF roughness',color='#1f77b4')
        axh.set_ylim()
        axh.set_xlabel('water level [m]')
        axh.set_ylabel('roughness [$\sqrt{m}$/s]')
        axh.legend()
        if MainChannel:
            titlestr = 'Water level dependent main channel roughness ' + \
                'at chainage ' + str(y)
        else:
            titlestr = 'Water level dependent floodplain1 roughness ' + \
                'at chainage ' + str(y)
        axh.set_title(titlestr)
        plt.grid()
        plt.tight_layout()
        if MainChannel:
            figtitlestr = 'main_roughness_chainage' + str(int(y))
        else:
            figtitlestr = 'fp1_roughness_chainage' + str(int(y))
        fig_name = '{}.png'.format(figtitlestr)
        fig_path = os.path.join(fig_dir, fig_name)
        plt.savefig(fig_path)
        plt.close('all')

    def __get_analytical_roughness(
            self, H_pos, b :float, y, MainChannel, case_name):
        tR_pos = []
        tH_pos = []
        for i in range(len(H_pos)):
            d = H_pos[i]-b
            if d >= 0:
                if MainChannel:
                    tH_pos.append(H_pos[i])
                    if case_name == 'case_01_rectangle':
                        #Cz = (150.0*d/(2.0*d+150))**(1/6)/0.03
                        Cz = (d)**(1/6)/0.03
                    elif case_name == 'case_05_dyke':
                        if d < 3.0:
                            Cz = (d)**(1/6)/0.03
                        else:
                            Cz = (d)**(1/6)/0.03
                    elif case_name == 'case_07_triangular':
                        if y > 500 and y < 9500:
                            if d < 2.0:
                                Cz = (d)**(1/6)/0.03
                            else:
                                Cz = (d)**(1/6)/0.03
                        else:
                            if d < 2.0:
                                Cz = (d)**(1/6)/0.03
                            else:
                                Cz = (d)**(1/6)/0.03
                    else:
                        if d < 2.0:
                            #Cz = (50.0*d/(2.0*d+50))**(1/6)/0.03
                            Cz = (d)**(1/6)/0.03
                        else:
                            #Cz = (50.0*d/(2.0*2+50))**(1/6)/0.03
                            Cz = (d)**(1/6)/0.03
                    tR_pos.append(Cz)
                else:  # floodplain1
                    if case_name == 'case_06_plassen':
                        if y >= 1420 and y <= 1580:
                            if d < 2.0:
                                Cz = ((-0.53+10)*30/((-0.53+10)*2+30))**(1/6)/0.07
                            elif d >= 2.0 and d < 2.0 + 0.01:  # +0.00048m in reality (filling lake completely)
                                Cz = ((30*(b+8))/(2*(b+8)+30))**(1/6)/0.07
                            else:
                                Cz = (100*(d-2.0)/(2.0*(d+48)))**(1/6)/0.07
                        else:
                            if d > 2.0 + 0.01:
                                Cz = (100*(d-2.0)/(2.0*(d+48)))**(1/6)/0.07
                        try:
                            tR_pos.append(Cz)
                            tH_pos.append(H_pos[i])
                        except NameError:
                            pass
                    if d > 2.0:
                        if case_name == 'case_02_compound':
                            #Cz = (100*(d-2.0)/(2.0*(d+48)))**(1/6)/0.07
                            Cz = (d-2.0)**(1/6)/0.07
                        elif case_name == 'case_03_threestage':
                            if d <= 2.5:
                                Cz = (d-2)**(1/6)/0.07
                            else:
                                #Cz = ((50*0.5+100*(d-2.5))/(51+2.0*d+45))**(1/6)/0.07
                                Cz = ((d-2.0)**(1/6)/0.07+(d-2.5)**(1/6)/0.07)/2
                        elif case_name == 'case_04_storage':
                            if y >= 1250 and y <= 1750:
                                Cz = (50*(d-2.0)/(2.0*(d+23)))**(1/6)/0.07
                            else:
                                Cz = (100*(d-2.0)/(2.0*(d+48)))**(1/6)/0.07
                        elif case_name == 'case_05_dyke':
                            th = 0.25  # transition height
                            if d > 3.0 and d <= 3.0+th:
                                Cz = ((100*(d-3.0) + 100/th*(d-3.0))/(4*(1.0/th)*(d-3.0) + 2*(d-3.0)+100))**(1/6)/0.07
                            elif d > 3.0+th:
                                Cz = ((100*(d-3.0)+100)/(2*(d-3.0)+100+4.0))**(1/6)/0.07
                        elif case_name == 'case_07_triangular':
                            if y > 500 or y < 9500:
                                Cz = (300*(d-2.0)/(2.0*(d+148)))**(1/6)/0.07
                            else:
                                Cz = (300*(d-2.0)/(2.0*(d+148)))**(1/6)/0.07
                        try:
                            tR_pos.append(Cz)
                            tH_pos.append(H_pos[i])
                        except NameError:
                            pass
        return tH_pos, tR_pos

    def __get_roughness_data(self, input_file):
        """[summary]
        
        Arguments:
            input_file {str} -- roughness csv file
        
        Returns:
            {tuple} --  Y = chainage; N = number of water levels;
                        H_pos = water level, R_pos = roughness (Cz)
        """
        # Reading roughness.csv file
        # Output lists ===
        
        hpos_tmp = []
        rpos_tmp = []
        
        S = []
        Y = []
        N = []
        assert os.path.exists(input_file), '' + \
            'Input file {} does not exist'.format(input_file)
        
        with open(input_file) as fin:
            y_prev = ''
            n_counter = 0
            for line in fin:
                ls = line.strip().split(',')
                if 'Name' in ls[0]:
                    section_index = ls.index('SectionType')
                    y_index = ls.index('Chainage')
                    hpos_index = ls.index('H_pos')
                    rpos_index = ls.index('R_pos__f(h)')
                else:
                    if ls[y_index] != y_prev:
                        Y.append(float(ls[y_index]))      # chainage
                        S.append(ls[section_index])  # Section name
                        if len(N) == 0:
                            if len(hpos_tmp) != 0:
                                H_pos = [hpos_tmp]
                                R_pos = [rpos_tmp]
                                N.append(n_counter)
                                n_counter = 0
                        else:
                            H_pos.append(hpos_tmp)      # water level
                            R_pos.append(rpos_tmp)      # roughness (Chezy)
                            N.append(n_counter)
                            n_counter = 0
                        hpos_tmp = []
                        rpos_tmp = []
                    hpos_tmp.append(float(ls[hpos_index]))
                    rpos_tmp.append(float(ls[rpos_index]))

                    y_prev = ls[y_index]
                    n_counter += 1          # number of data at each chainage
        H_pos.append(hpos_tmp) 
        R_pos.append(rpos_tmp)
        N.append(n_counter)
        return (S, Y, N, H_pos, R_pos)
