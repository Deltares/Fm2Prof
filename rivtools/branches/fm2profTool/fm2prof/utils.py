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


"""
output is (x, y, id, vaklengte) + (, branch, offset) ??

vaklengte = di+1/2 + di-1/2

"""

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

				
				# Append all poitns
				x.extend(xtmp)
				y.extend(ytmp)
				cid.extend(cidtmp)
				coff.extend(cofftmp)
				cdis.extend(cdistmp)


	with open(crossectionlocationfile, 'w') as f:
		for i in range(len(x)):
			f.write('{}, {:.4f}, {:.4f}, {:.2f}, {}, {:.2f}\n'.format(cid[i], x[i], y[i], cdis[i], bid[i], coff[i]))







