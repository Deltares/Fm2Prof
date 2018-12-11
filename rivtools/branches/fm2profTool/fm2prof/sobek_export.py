import locale
import numpy as np
import collections

from fm2prof import Functions as FE

def geometry_to_csv(cross_sections, chainages, file_path):
    with open(file_path, 'w') as f:
        # write header
        f.write('id,Name,Data_type,level,Total width,Flow width,Profile_type,branch,chainage,width main channel,width floodplain 1,width floodplain 2,width sediment transport,Use Summerdike,Crest level summerdike,Floodplain baselevel behind summerdike,Flow area behind summerdike,Total area behind summerdike,Use groundlayer,Ground layer depth\n')

        for index, section in enumerate(cross_sections):
            if chainages is not None:
                chainage = chainages[index]
            else:
                chainage = None

            _write_geometry(f, section, chainage)

def roughness_to_csv(cross_sections, chainages, file_path):
    with open(file_path, 'w') as f:
        # write header
        f.write('Name,Chainage,RoughnessType,SectionType,Dependance,Interpolation,Pos/neg,R_pos_constant,Q_pos,R_pos_f(Q),H_pos,R_pos__f(h),R_neg_constant,Q_neg,R_neg_f(Q),H_neg,R_neg_f(h)\n')

        try:

            for index, section in enumerate(cross_sections):
                if chainages is not None:
                    chainage = chainages[index]
                else:
                    chainage = None

                _write_roughness(f, section, 'alluvial', chainage)

            for index, section in enumerate(cross_sections):
                if chainages is not None:
                    chainage = chainages[index]
                else:
                    chainage = None

                _write_roughness(f, section, 'nonalluvial', chainage)
        except IndexError:
            return None

def _write_geometry(write_object, cross_section, chainage=None):
    # write meta
    # note, the chainage is currently set to the X-coordinate of the cross-section (straight channel)
    # note, the channel naming strategy must be discussed, currently set to 'Channel' for all cross-sections
    if chainage is None:
        chainage = cross_section.location[0]
    else:
        chainage = float(chainage)

    total_width = cross_section.total_width[-1]

    b_summerdike = '0'
    crest_level = ''
    floodplain_base = ''
    total_area = ''

    if cross_section.extra_total_volume > 0:
        b_summerdike = '1'
        crest_level = str(cross_section.crest_level)
        total_area = str(cross_section.total_area)

        # check for nan, because a channel with only one roughness value (ideal case) will not have this value
        if np.isnan(cross_section.floodplain_base) == False:
            floodplain_base = str(cross_section.floodplain_base)

    write_object.write(cross_section.name + ',,' + 'meta' + ',,,,' + 'ZW' + ',' + 'Channel1' + ',' + str(chainage) + ',' + str(cross_section.alluvial_width) + ',' + str(cross_section.nonalluvial_width) + ',,,' + b_summerdike + ',' + crest_level + ',' + floodplain_base + ',' + total_area + ',' + total_area + ',,,,,,' + '\n')

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
        write_object.write(cross_section.name + ',,' + 'geom' + ',' + z_format.format(z_value[index]) + ',' + str(width) + ',' + str(flow_width) + ',,,,,,,,,,,,,,' + '\n')

def _write_roughness(write_object, cross_section, type, chainage=None):
    # note, the chainage is currently set to the X-coordinate of the cross-section (straight channel)
    # note, the channel naming strategy must be discussed, currently set to 'Channel' for all cross-sections
    if chainage is None:
        chainage = cross_section.location[0]
    else:
        chainage = float(chainage)

    waterlevels = cross_section.alluvial_friction_table[0]

    # round off to 2 decimals
    waterlevels = np.ceil(waterlevels * 100) / 100

    if type == 'alluvial':
        table = cross_section.alluvial_friction_table
        plain = 'Main'
    elif type == 'nonalluvial':
        table = cross_section.nonalluvial_friction_table
        plain = 'FloodPlain1'
    else:
        raise Exception('choose either alluvial or nonalluvial')

    for index, level in enumerate(waterlevels):
        chezy = table[1].iloc[index]

        if np.isnan(chezy) == False:
            write_object.write('Channel1' + ',' + str(chainage) + ',' + 'Chezy' + ',' + plain + ',' + 'Waterlevel' + ',' + 'Linear' + ',' + 'Same' + ',,,,' + str(level) + ',' + str(chezy) + ',,,,,' + '\n')