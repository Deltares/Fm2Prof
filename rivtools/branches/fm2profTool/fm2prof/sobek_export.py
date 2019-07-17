""" Input/output """

import locale
import numpy as np
import collections
from fm2prof import Functions as FE



def export_geometry(cross_sections, file_path, fmt='sobek3'):
    with open(file_path, 'w') as f:
        if fmt == 'sobek3':
            """ SOBEK 3 style csv """
            _write_geometry_sobek3(f, cross_sections)
        elif fmt == 'dflow1d':
            """ DFM 1D style """
            _write_geometry_fm1d(f, cross_sections)
        elif fmt == 'testformat':
            """ test format for system tests, only has geometry (no summerdike) """
            _write_geometry_testformat(f, cross_sections)

def export_roughness(cross_sections, file_path, fmt='sobek3'):
    with open(file_path, 'w') as f:
        if fmt == 'sobek3':
            """ SOBEK 3 style csv """
            _write_roughness_sobek3(f, cross_sections)
        elif fmt == 'fm1d':
            """ DFM 1D style """
            _write_roughness_fm1d(f, cross_sections)
        elif fmt == 'testformat':
            """ test format for system tests, only has geometry (no summerdike) """
            _write_roughness_testformat(f, cross_sections)

def export_volumes(cross_sections, file_path):
    """Write to file the volume/waterlevel information"""

    with open(file_path, 'w') as f:
        # Write header
        f.write("id,z,2D_total_volume,2D_flow_volume,2D_wet_area,2D_flow_area,1D_total_volume_sd,1D_total_volume,1D_flow_volume_sd,1D_flow_volume,1D_total_width,1D_flow_width\n")

        for css in cross_sections:
            for i in range(len(css._css_z)):
                outputdata = dict(id=css.name,
                                  z=css._css_z[i],
                                  tv2d=css._fm_total_volume[i],
                                  fv2d=css._fm_flow_volume[i],
                                  wa2d=css._fm_wet_area[i],
                                  fa2d=css._fm_flow_area[i],
                                  tvsd1d=css._css_total_volume_corrected[i],
                                  tv1d=css._css_total_volume[i],
                                  fvsd1d=css._css_flow_volume_corrected[i],
                                  fv1d=css._css_flow_volume[i],
                                  tw1d=css._css_total_width[i],
                                  fw1d=css._css_flow_width[i])
                
                f.write('{id},{z},{tv2d},{fv2d},{wa2d},{fa2d},{tvsd1d},{tv1d},{fvsd1d},{fv1d},{tw1d},{fw1d}\n'.format(**outputdata))
        ## Construct array and append to file
        #for css in cross_sections:
        #    for i in range(len(css._css_z)):
        #        try:
        #            f.write('{}, {}, {}, {}\n'.format(css.name, 
        #                                              css._css_z[i], 
        #                                              css._css_total_volume_corrected[i], 
        #                                              css._fm_total_volume[i]))
        #        except TypeError:
        #            f.write('{}, {}, {}, {}\n'.format(css.name, 
        #                                              css._css_z[i], 
        #                                              css._css_total_volume[i], 
        #                                              css._fm_total_volume[i]))
                        

def export_crossSectionLocations(cross_sections, file_path):
    """ DIMR format """
    with open(file_path, 'w') as fid:
        # Write general secton
        fid.write('[General]\nmajorVersion\t\t\t= 1\nminorversion\t\t\t= 0\nfileType\t\t\t\t= crossLoc\n\n') 

        for css in cross_sections:
            fid.write('[CrossSection]\n')
            fid.write('\tid\t\t\t\t\t= {}\n'.format(css.name))
            fid.write('\tbranchid\t\t\t= {}\n'.format(css.branch))
            fid.write('\tchainage\t\t\t= {}\n'.format(css.chainage))
            fid.write('\tshift\t\t\t\t= 0.000\n')
            fid.write('\tdefinition\t\t\t= {}\n\n'.format(css.name))


""" test file formats """
def _write_geometry_testformat(fid, cross_sections):
    # write header
    fid.write('chainage,z,total width,flow width\n')
    for index, cross_section in enumerate(cross_sections):
        for i in range(len(cross_section.z)):
            fid.write('{}, {}, {}, {}\n'.format(cross_section.chainage,
                                                cross_section.z[i],
                                                cross_section.total_width[i],
                                                cross_section.flow_width[i]))

def _write_roughness_testformat(fid, cross_sections):
    # write header
    fid.write('chainage,type,waterlevel,chezy roughness\n')

    for roughnesstype in ('alluvial', 'nonalluvial'):
        for index, cross_section in enumerate(cross_sections):
            waterlevels = cross_section.alluvial_friction_table[0]
            
            if roughnesstype == 'alluvial':
                table = cross_section.alluvial_friction_table
            elif roughnesstype == 'nonalluvial':
                table = cross_section.nonalluvial_friction_table
                

            for index, level in enumerate(waterlevels):
                try:
                    chezy = table[1].iloc[index]
                except IndexError:
                    break
                if np.isnan(chezy) == False:
                    fid.write('{}, {}, {}, {}\n'.format(cross_section.chainage, roughnesstype, level, chezy))


""" FM 1D file formats """
def _write_geometry_fm1d(fid, cross_sections):
    """ FM1D uses a configuration 'Delft' file style format """
    
    # Write general secton
    fid.write('[General]\nmajorVersion\t\t\t= 1\nminorversion\t\t\t= 0\nfileType\t\t\t\t= crossDef\n\n')

    for index, css in enumerate(cross_sections):
        z = ["{:.4f}".format(iz) for iz in css.z]
        fw = ["{:.4f}".format(iz) for iz in css.flow_width]
        tw = ["{:.4f}".format(iz) for iz in css.total_width]

        # check for nan, because a channel with only one roughness value (ideal case) will not have this value
        if np.isnan(css.floodplain_base) == False:
            floodplain_base = str(css.floodplain_base)
        else:
            floodplain_base = str(css.crest_level)

        fid.write("[Definition]\n")
        fid.write("\tid\t\t\t\t\t= {}\n".format(css.name) +\
                "\ttype\t\t\t\t= tabulated\n" +\
                "\tthalweg\t\t\t\t= 0.000\n" +\
                "\tnumLevels\t\t\t= {}\n".format(len(z)) +\
                "\tlevels\t\t\t\t= {}\n".format(' '.join(z))+\
                "\tflowWidths\t\t\t= {}\n".format(' '.join(fw))+\
                "\ttotalWidths\t\t\t= {}\n".format(' '.join(tw))+\
                "\tsd_crest\t\t\t= {:.4f}\n".format(css.crest_level)+\
                "\tsd_flowArea\t\t\t= {}\n".format(css.extra_flow_area)+\
                "\tsd_totalArea\t\t= {:.4f}\n".format(css.extra_total_area)+\
                "\tsd_baseLevel\t\t= {}\n".format(css.floodplain_base)+\
                "\tmain\t\t\t\t= {}\n".format(css.alluvial_width)+\
                "\tfloodPlain1\t\t\t= {}\n".format(css.nonalluvial_width)+\
                "\tfloodPlain2\t\t\t= 0.0\n"+\
                "\tgroundlayerUsed\t\t= 0\n"+\
                "\tgroundLayer\t\t\t= 0.000\n\n"
                )

def _write_roughness_fm1d(fid, cross_sections):
    raise NotImplementedError()

""" SOBEK 3 file formats """
def _write_geometry_sobek3(fid, cross_sections):
    # write meta
    # note, the chainage is currently set to the X-coordinate of the cross-section (straight channel)
    # note, the channel naming strategy must be discussed, currently set to 'Channel' for all cross-sections
    
    # write header
    fid.write('id,Name,Data_type,level,Total width,Flow width,Profile_type,branch,chainage,width main channel,width floodplain 1,width floodplain 2,width sediment transport,Use Summerdike,Crest level summerdike,Floodplain baselevel behind summerdike,Flow area behind summerdike,Total area behind summerdike,Use groundlayer,Ground layer depth\n')

    for index, cross_section in enumerate(cross_sections):
        total_width = cross_section.total_width[-1]

        b_summerdike = '0'
        crest_level = ''
        floodplain_base = ''
        total_area = ''

        if cross_section.extra_total_volume > 0:
            b_summerdike = '1'
            crest_level = str(cross_section.crest_level)
            total_area = str(cross_section.extra_total_area)

            # check for nan, because a channel with only one roughness value (ideal case) will not have this value
            if np.isnan(cross_section.floodplain_base) == False:
                floodplain_base = str(cross_section.floodplain_base)
            else:
                floodplain_base = str(cross_section.crest_level) #virtual summer dike

        fid.write(cross_section.name + ',,' + 'meta' + ',,,,' + 'ZW' + ',' + str(cross_section.branch) + ',' + str(cross_section.chainage) + ',' + str(cross_section.alluvial_width) + ',' + str(cross_section.nonalluvial_width) + ',,,' + b_summerdike + ',' + crest_level + ',' + floodplain_base + ',' + total_area + ',' + total_area + ',,,,,,' + '\n')

        # this is to avoid the unique z-value error in sobek, the added 'error' depends on the total_width, this is to make sure the order or points is correct
        z_format = '{:.8f}'
        increment = np.array(range(1, cross_section.z.size + 1)) * 1e-5
        z_value = cross_section.z + increment

        # sort z_value
        # TODO: CHECK (this was due to an old bug, check whether still applicable)
        # z_value = np.sort(z_value)

        # write geometry information
        for index, width in enumerate(cross_section.total_width):
            flow_width = cross_section.flow_width[index]
            fid.write(cross_section.name + ',,' + 'geom' + ',' + z_format.format(z_value[index]) + ',' + str(width) + ',' + str(flow_width) + ',,,,,,,,,,,,,,' + '\n')

def _write_roughness_sobek3(fid, cross_sections):
    # note, the chainage is currently set to the X-coordinate of the cross-section (straight channel)
    # note, the channel naming strategy must be discussed, currently set to 'Channel' for all cross-sections
    
    # write header
    fid.write('Name,Chainage,RoughnessType,SectionType,Dependance,Interpolation,Pos/neg,R_pos_constant,Q_pos,R_pos_f(Q),H_pos,R_pos__f(h),R_neg_constant,Q_neg,R_neg_f(Q),H_neg,R_neg_f(h)\n')

    for roughnesstype in ('alluvial', 'nonalluvial'):
        for index, cross_section in enumerate(cross_sections):
            waterlevels = cross_section.alluvial_friction_table[0]

            # round off to 2 decimals
            waterlevels = np.ceil(waterlevels * 100) / 100

            if roughnesstype == 'alluvial':
                table = cross_section.alluvial_friction_table
                plain = 'Main'
            elif roughnesstype == 'nonalluvial':
                table = cross_section.nonalluvial_friction_table
                plain = 'FloodPlain1'
            else:
                raise Exception('choose either alluvial or nonalluvial')

            for index, level in enumerate(waterlevels):
                try:
                    chezy = table[1].iloc[index]
                except IndexError:
                    break

                if np.isnan(chezy) == False:
                    fid.write(str(cross_section.branch) + ',' + str(cross_section.chainage) + ',' + 'Chezy' + ',' + plain + ',' + 'Waterlevel' + ',' + 'Linear' + ',' + 'Same' + ',,,,' + str(level) + ',' + str(chezy) + ',,,,,' + '\n')


           