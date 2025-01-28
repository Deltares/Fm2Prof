from __future__ import annotations

from pathlib import Path

import numpy as np

from fm2prof.common import FM2ProfBase


class GenerateCrossSectionLocationFile(FM2ProfBase):
    """Build a cross-section input file for FM2PROF from a SOBEK 3 DIMR network definition file.

    The distance between cross-section is computed from the differences between the offsets/chainages.
    The beginning and end point of each branch are treated as half-distance control volumes.

    It supports an optional :ref:`branchRuleFile.

    Use as a function i.e. this code will generate a cross-section location file:

    >>> GenerateCrossSectionLocationFile(**input)

    Parameters
    ----------
        networkdefinitionfile: path to NetworkDefinitionFile.ini

        crossectionlocationfile: path to the desired output file

        branchrulefile: OPTIONAL path to a branchrulefile




    branchrulefile
    ^^^^^^^^^^^^^^
    This file may be used to exclude certain computational points from being
    used as the location of a cross-section. This is particularily useful
    when smaller branches connect to a major branch.

    The branchrule file is a comma-seperates file with the following syntaxt:

    .. code-block:: shell

        branch,rules

    Here, `branch` is the name of the branch and `rules` are rules for exclusion

    Supported general rules are:

    - onlyFirst: only keep the first cross-section, and exclude all others
    - onlyLast: only keep the last cross-section, and exclude all others
    - onlyEdges: only keep the first and last cross-section, and exclude all others
    - ignoreFirst: exclude the first cross-section on a branch
    - ignoreLast: exclude the last cross-section on a branch
    - ignoreEdges: exclude the first and last cross-section on a branch
    - noRule: use to not use any of the above rules

    Additionally, specific cross-sections can be excluded by id. For example:


    .. code-block:: shell

        Channel1, noRule, channel_1_350.000

    In this case, the computational point with name `channel_1_350.000` will
    not be used as the location of a cross-section.

    Rules and individual exclusions can be mixed, e.g.:

    .. code-block:: shell

        Channel1, ignoreLast, channel_1_350.000

    """

    def __init__(
        self,
        network_definition_file: str | Path,
        cross_section_location_file: str | Path,
        branch_rule_file: str | Path = "",
    ) -> None:
        """Generate cross section location file object.

        Args:
            network_definition_file (str | Path): network definition file
            crossection_location_file (str | Path): crosssection location file
            branchrule_file (str | Path, optional): . Defaults to "".

        """
        super().__init__()

        network_definition_file, cross_section_location_file, branch_rule_file = map(
            Path,
            [network_definition_file, cross_section_location_file, branch_rule_file],
        )

        if not network_definition_file.exists():
            err_msg = "Network difinition file not found"
            raise FileNotFoundError(err_msg)

        self._network_definition_file_to_input(network_definition_file, cross_section_location_file, branch_rule_file)

    def _parse_network_definition_file(self, network_definition_file: Path, branchrules: dict | None = None) -> dict:
        """Parse network definition file.

        Output:

        x,y : coordinates of cross-section
        cid : name of the cross-section
        cdis: half-way distance between cross-section points on either side
        bid : name of the branch
        coff:  chainage of cross-section on branch

        """
        if not branchrules:
            branchrules = {}

        # Open network definition file, for each branch extract necessary info
        x = []  # x-coordinate of cross-section centre
        y = []  # y-coordinate of cross-section centre
        cid = []  # id of cross-section
        bid = []  # id of 1D branch
        coff = []  # offset of cross-section on 1D branch ('chainage')
        cdis = []  # distance of 1D branch influenced by crosss-section ('vaklengte')

        with network_definition_file.open("r") as f:
            for line in f:
                if line.strip().lower() == "[branch]":
                    branchid = f.readline().split("=")[1].strip()
                    for _ in range(10):
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
                            cdistmp = np.append(np.diff(cofftmp) / 2, [0]) + np.append([0], np.diff(cofftmp) / 2)

                    cdistmp = list(cdistmp)
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
                        rule = branchrules[branchid].get("rule")
                        exceptions = branchrules[branchid].get("exceptions")
                        if rule:
                            (
                                xtmp,
                                ytmp,
                                cidtmp,
                                cdistmp,
                                bidtmp,
                                cofftmp,
                            ) = self._apply_branch_rules(rule, xtmp, ytmp, cidtmp, cdistmp, bidtmp, cofftmp)
                        if exceptions:
                            (
                                xtmp,
                                ytmp,
                                cidtmp,
                                cdistmp,
                                bidtmp,
                                cofftmp,
                            ) = self._apply_branch_exceptions(exceptions, xtmp, ytmp, cidtmp, cdistmp, bidtmp, cofftmp)
                        c = len(xtmp)
                        for ic in xtmp, ytmp, cidtmp, cdistmp, bidtmp, cofftmp:
                            if len(ic) != c:
                                raise ValueError

                    # Append all points
                    x.extend(xtmp)
                    y.extend(ytmp)
                    cid.extend(cidtmp)
                    cdis.extend(cdistmp)
                    bid.extend(bidtmp)
                    coff.extend(cofftmp)
        return {"x": x, "y": y, "css_id": cid, "css_len": cdis, "branch_id": bid, "css_offset": coff}

    def _network_definition_file_to_input(
        self,
        network_definition_file: Path,
        crossection_location_file: Path,
        branchrule_file: Path,
    ) -> None:
        branchrules: dict = {}

        if branchrule_file.is_file():
            branchrules = self._parse_branch_rule_file(branchrule_file)

        network_dict = self._parse_network_definition_file(network_definition_file, branchrules)

        self._write_cross_section_location_file(crossection_location_file, network_dict)

    def _apply_branch_exceptions(  # noqa: PLR0913
        self,
        exceptions: list[str],
        x: list[float],
        y: list[float],
        cid: list[str],
        cdis: list[float],
        bid: list[str],
        coff: list[float],
    ) -> tuple[list[float], list[float], list[str], list[float], list[str], list[float]]:
        for exc in exceptions:
            if exc not in cid:
                self.set_logger_message(f"{exc} not found in branch", "error")
                continue

        pop_indices = [cid.index(exc) for exc in exceptions]

        for pop_index in sorted(pop_indices, reverse=True):
            if pop_index == 0:
                (
                    x,
                    y,
                    cid,
                    cdis,
                    bid,
                    coff,
                ) = self._apply_branch_rules("ignorefirst", x, y, cid, cdis, bid, coff)
            elif pop_index == len(x) - 1:
                (
                    x,
                    y,
                    cid,
                    cdis,
                    bid,
                    coff,
                ) = self._apply_branch_rules("ignorelast", x, y, cid, cdis, bid, coff)
            else:
                # the distance of the popped value is divided over the two on aither side.
                cdis[pop_index - 1] += cdis[pop_index] / 2
                cdis[pop_index + 1] += cdis[pop_index] / 2

                # then, pop the value
                for v in [x, y, cid, cdis, bid, coff]:
                    v.pop(pop_index)

        return x, y, cid, cdis, bid, coff

    def _apply_branch_rules(  # noqa: PLR0913
        self,
        rule: str,
        x: float,
        y: float,
        cid: str,
        cdis: float,
        bid: str,
        coff: float,
    ) -> tuple[
        list[float],
        list[float],
        list[str],
        list[float],
        list[str],
        list[float],
    ]:
        # bfunc: what points to pop (remove from list)
        bfunc = {
            "norule": lambda x: x,
            "onlyedges": lambda x: [
                x[0],
                x[-1],
            ],  # only keep the 2 cross-section on either end of the branch
            "ignoreedges": lambda x: x[1:-1],  # keep everything except 2 css on either end of the branch
            "ignorelast": lambda x: x[:-1],  # keep everything except last css on branch
            "ignorefirst": lambda x: x[1:],  # keep everything except first css on branch
            "onlyfirst": lambda x: [x[0]],  # keep only the first css on branch
            "onlylast": lambda x: [x[-1]],  # keep only the last css on branch
        }
        # disfunc: how to modify lengths
        disfunc = {
            "onlyedges": lambda x: [sum(x) / 2] * 2,
            "ignoreedges": lambda x: [sum(x[:2]), *x[2:-2], sum(x[-2:])],
            "ignorelast": lambda x: [*x[:-2], sum(x[-2:])],
            "ignorefirst": lambda x: [sum(x[:2]), *x[2:]],
            "onlyfirst": lambda x: [sum(x)],
            "onlylast": lambda x: [sum(x)],
            "norule": lambda x: x,
        }

        try:
            bf = bfunc[rule.lower().strip()]
            disf = disfunc[rule.lower().strip()]
            return bf(x), bf(y), bf(cid), disf(cdis), bf(bid), bf(coff)
        except KeyError:
            self.set_logger_message(
                f"'{rule}' is not a known branchrules. Known rules are: {list(bfunc.keys())}",
                "error",
            )

    def _parse_branch_rule_file(self, branchrulefile: Path, delimiter: str = ",") -> dict[str, dict]:
        """Parse the branchrule file which is a delimited file (comma by default)."""
        branchrules: dict = {}
        with branchrulefile.open("r") as f:
            lines = [line.strip().split(delimiter) for line in f if len(line) > 1]

        for line in lines:
            branch: str = line[0].strip()
            rule: str = line[1].strip()
            exceptions: list = []
            if len(line) > 2:  # noqa: PLR2004
                exceptions = [e.strip() for e in line[2:]]

            branchrules[branch] = {"rule": rule, "exceptions": exceptions}

        return branchrules

    def _write_cross_section_location_file(self, crossectionlocationfile: Path, network_dict: dict) -> None:
        """Write cross section location file.

        List inputs:

        x,y : coordinates of cross-section
        cid : name of the cross-section
        cdis: half-way distance between cross-section points on either side
        bid : name of the branch
        coff:  chainage of cross-section on branch
        """
        x = network_dict.get("x")
        y = network_dict.get("y")
        cid = network_dict.get("css_id")
        cdis = network_dict.get("css_len")
        bid = network_dict.get("branch_id")
        coff = network_dict.get("css_offset")

        with crossectionlocationfile.open("w") as f:
            f.write("name,x,y,length,branch,offset\n")
            for i in range(len(x)):
                f.write(f"{cid[i]}, {x[i]:.4f}, {y[i]:.4f}, {cdis[i]:.2f}, {bid[i]}, {coff[i]:.2f}\n")
