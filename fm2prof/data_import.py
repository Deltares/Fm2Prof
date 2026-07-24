"""Contains code for importing files to fm2prof."""

from __future__ import annotations

# import from standard library
from pathlib import Path

# import from dependencies
import numpy as np

# import from  package
from fm2prof.common import FM2ProfBase


class ImportInputFiles(FM2ProfBase):
    """Contains all functions related to the import of files."""

    def css_file(self, file_path: Path | str, delimiter: str = ",") -> dict:
        """Read the cross-section location file."""
        skip_line = False  # flag to skip line if file has header

        if not file_path or not Path(file_path).exists():
            err_msg = f"No file path for Cross Section location file was given, or could not be found at {file_path}"
            raise OSError(err_msg)

        with Path(file_path).open("r") as fid:
            input_data = {"xy": [], "id": [], "branchid": [], "length": [], "chainage": []}
            for lineno, line in enumerate(fid):
                try:
                    (cssid, x, y, length, branchid, chainage) = line.split(delimiter)
                except ValueError:
                    # revert to legacy format
                    (x, y, branchid, length, chainage) = line.split(delimiter)
                    cssid = branchid + "_" + str(round(float(chainage)))
                try:
                    float(x)
                except ValueError:
                    if lineno == 0:
                        # file has header. Skip header and try again next
                        skip_line = True
                if not skip_line:
                    input_data["xy"].append((float(x), float(y)))
                    input_data["id"].append(cssid)
                    input_data["length"].append(float(length))
                    input_data["branchid"].append(branchid.strip())
                    input_data["chainage"].append(float(chainage))
                skip_line = False

            # Convert everything to ndarray
            for key in input_data:
                input_data[key] = np.array(input_data[key])
            return input_data
