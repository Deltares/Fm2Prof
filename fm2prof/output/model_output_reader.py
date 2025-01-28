"""Model output data reader."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import tqdm
from netCDF4 import Dataset

from fm2prof.common import FM2ProfBase

if TYPE_CHECKING:
    from logging import Logger


class ModelOutputReader(FM2ProfBase):
    """Provide methods to post-process 1D and 2D data.

    The data is prost-processed by writing csv files of output locations (observation stations)
    that are both in 1D and 2D. It produces two csv files that
    are input for :meth:`fm2prof.utils.ModelOutputPlotter`

    Example use:

    >>> # initialise & give path to 1D, 2D model resusts
    >>> from fm2prof.utils import ModelOutputReader
    >>> output = ModelOutputReader()
    >>> output.path_flow1d = path_to_dimr_directory
    >>> output.path_flow2d = path_to_nc_file
    >>> # Read and write 1d output to csv
    >>> output.load_flow1d_data()
    >>> #
    >>> output.get_1d2d_map()
    >>> output.load_flow2d_data()
    """

    __fileOutName_F1D_Q = "1D_Q.csv"
    __fileOutName_F1D_H = "1D_H.csv"
    __fileOutName_F2D_Q = "2D_Q.csv"
    __fileOutName_F2D_H = "2D_H.csv"

    _key_1d_q_name = "observation_id"
    _key_1d_q = "water_discharge"
    _key_1d_time = "time"
    _key_1d_h_name = "observation_id"
    _key_1d_h = "water_level"

    _key_2d_q_name = "cross_section_name"
    _key_2d_q = "cross_section_discharge"
    _key_2d_time = "time"
    _key_2d_h_name = "station_name"
    _key_2d_h = "waterlevel"
    __fileOutName_1D2DMap = "map_1d_2d.csv"

    _time_fmt = "%Y-%m-%d %H:%M:%S"

    def __init__(
        self,
        logger: Logger | None = None,
        start_time: datetime | None = None,
        stop_time: datetime | None = None,
    ) -> None:
        """Instantiate a ModelOutputReader object.

        Args:
            logger (Logger | None, optional): logger. Defaults to None.
            start_time (datetime | None, optional): start time. Defaults to None.
            stop_time (datetime | None, optional): stop time. Defaults to None.

        """
        super().__init__(logger=logger)

        self._path_out: Path = Path()
        self._path_flow1d: Path = Path()
        self._path_flow2d: Path = Path()

        self._data_1d_q: pd.DataFrame = None
        self._time_offset_1d: int = 0

        self._start_time: datetime | None = start_time
        self._stop_time: datetime | None = stop_time

    @property
    def start_time(self) -> datetime | None:
        """If defined, used to mask data."""
        return self._start_time

    @start_time.setter
    def start_time(self, input_time: datetime) -> None:
        if isinstance(input_time, datetime):
            self._start_time = input_time

    @property
    def stop_time(self) -> datetime | None:
        """If defined, used to mask data."""
        return self._stop_time

    @stop_time.setter
    def stop_time(self, input_time: datetime) -> None:
        if isinstance(input_time, datetime):
            self._stop_time = input_time

    @property
    def path_flow1d(self) -> Path:
        """Return path to flow 1D file."""
        return self._path_flow1d

    @path_flow1d.setter
    def path_flow1d(self, path: Path | str) -> None:
        # Verify path is dir
        if not Path(path).is_file():
            err_msg = f"Given path, {path}, is not a file."
            raise ValueError(err_msg)
        # set attribute
        self._path_flow1d = Path(path)

    @property
    def path_flow2d(self) -> Path:
        """Return path to flow 2D file."""
        return self._path_flow2d

    @path_flow2d.setter
    def path_flow2d(self, path: Path | str) -> None:
        # Verify path is file
        if not Path(path).is_file():
            err_msg = f"Given path, {path}, is not a file."
            raise ValueError(err_msg)
        # set attribute
        self._path_flow2d = Path(path)

    def load_flow1d_data(self) -> None:
        """Load 'observations.nc' and outputs to csv file.

        .. note::
            Path to the 1D model must first be set by using
            >>> ModelOutputReader.path_flow1d = path_to_dir_that_contains_dimr_xml
        """
        if self.file_1d_q.is_file() & self.file_1d_h.is_file():
            self._data_1d_q = pd.read_csv(
                self.file_1d_q,
                index_col=0,
                parse_dates=True,
                date_format=self._time_fmt,
            )
            self._data_1d_h = pd.read_csv(
                self.file_1d_h,
                index_col=0,
                parse_dates=True,
                date_format=self._time_fmt,
            )
            self.set_logger_message("Using existing flow1d csv files")
        else:
            self.set_logger_message("Importing from NetCDF")
            self._data_1d_h, self._data_1d_q = self._import_1d_observations()
            self.set_logger_message("Writing to CSV (waterlevels)")
            self._data_1d_h.to_csv(self.file_1d_h)
            self.set_logger_message("Writing to CSV (discharge)")
            self._data_1d_q.to_csv(self.file_1d_q)

    def load_flow2d_data(self) -> None:
        """Load 2D output file.

        netCDF, must contain observation point results,
        matches to 1D result, output to csv

        .. note::
            Path to the 2D model output
            >>> ModelOutputReader.path_flow2d = path_to_netcdf_file
        """
        if self.file_2d_q.is_file() & self.file_2d_h.is_file():
            self._data_2d_q = pd.read_csv(
                self.file_2d_q,
                index_col=0,
                parse_dates=True,
                date_format=self._time_fmt,
            )
            self._data_2d_h = pd.read_csv(
                self.file_2d_h,
                index_col=0,
                parse_dates=True,
                date_format=self._time_fmt,
            )
            self.set_logger_message("Using existing flow2d csv files")
        else:
            # write to file
            self._import_2d_observations()

            # then load
            self._data_2d_q = pd.read_csv(
                self.file_2d_q,
                index_col=0,
                parse_dates=True,
                date_format=self._time_fmt,
            )
            self._data_2d_h = pd.read_csv(
                self.file_2d_h,
                index_col=0,
                parse_dates=True,
                date_format=self._time_fmt,
            )

    def get_1d2d_map(self) -> None:
        """Write a map between stations in 1D and stations in 2D.

        Matches based on identical characters in first nine slots
        """
        if self.file_1d2d_map.is_file():
            self.set_logger_message("using existing 1d-2d map")
            return
        self._get_1d2d_map()

    def read_all_data(self) -> None:
        """Read all data."""
        self.load_flow1d_data()
        self.get_1d2d_map()
        self.load_flow2d_data()

    @property
    def output_path(self) -> Path:
        """Return output path."""
        return self._path_out

    @output_path.setter
    def output_path(self, new_path: Path | str) -> None:
        newpath = Path(new_path)
        if newpath.is_dir():
            self._path_out = newpath
        else:
            err_msg = f"{new_path} is not a directory"
            raise ValueError(err_msg)

    @property
    def file_1d_q(self) -> Path:
        """Return path to 1D water discharge file."""
        return self.output_path.joinpath(self.__fileOutName_F1D_Q)

    @property
    def file_1d_h(self) -> Path:
        """Return path to 1D water level file."""
        return self.output_path.joinpath(self.__fileOutName_F1D_H)

    @property
    def file_2d_q(self) -> Path:
        """Return path to 2D discharge file."""
        return self.output_path.joinpath(self.__fileOutName_F2D_Q)

    @property
    def file_2d_h(self) -> Path:
        """Return path to 2D water level."""
        return self.output_path.joinpath(self.__fileOutName_F2D_H)

    @property
    def file_1d2d_map(self) -> Path:
        """Return path to 1D2D map file."""
        return self.output_path.joinpath(self.__fileOutName_1D2DMap)

    @property
    def data_1d_h(self) -> pd.DataFrame:
        """Apply start stop time to 1D water level data."""
        return self._apply_startstop_time(self._data_1d_h)

    @property
    def data_2d_h(self) -> pd.DataFrame:
        """Apply start stop time to 2D water level data."""
        return self._apply_startstop_time(self._data_2d_h)

    @property
    def data_1d_q(self) -> pd.DataFrame:
        """Apply start stop time to 1D discharge data."""
        return self._apply_startstop_time(self._data_1d_q)

    @property
    def data_2d_q(self) -> pd.DataFrame:
        """Apply start stop time to 2D discharge data."""
        return self._apply_startstop_time(self._data_2d_q)

    @property
    def time_offset_1d(self) -> int:
        """Return time offset for 1D data."""
        return self._time_offset_1d

    @time_offset_1d.setter
    def time_offset_1d(self, seconds: int = 0) -> None:
        self._time_offset_1d = seconds

    def _apply_startstop_time(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply stop/start time to data."""
        if self.stop_time is None:
            self.stop_time = data.index[-1]
        if self.start_time is None:
            self.start_time = data.index[0]

        if self.start_time >= self.stop_time:
            err_msg = "Stop time ({self.stop_time}) should be later than start time ({self.start_time})"
            self.set_logger_message(
                err_msg,
                "error",
            )
            raise ValueError(err_msg)
        if bool(self.start_time) and (self.start_time >= data.index[-1]):
            err_msg = f"Provided start time {self.start_time} is later than last record in data ({data.index[-1]})"
            self.set_logger_message(
                err_msg,
                "error",
            )
            raise ValueError(err_msg)
        if bool(self.stop_time) and (self.stop_time <= data.index[0]):
            err_msg = f"Provided stop time {self.stop_time} is earlier than first record in data ({data.index[0]})"
            self.set_logger_message(
                err_msg,
                "error",
            )
            raise ValueError(err_msg)

        if bool(self.start_time) and bool(self.stop_time):
            return data[(data.index >= self.start_time) & (data.index <= self.stop_time)]
        if bool(self.start_time) and not bool(self.stop_time):
            return data[(data.index >= self.start_time)]
        if not bool(self.start_time) and bool(self.stop_time):
            return data[data.index <= self.stop_time]
        return data

    @staticmethod
    def _parse_names(nclist: list[str], encoding: str = "utf-8") -> list[str]:
        """Parse the bytestring list of names in netcdf."""
        return ["".join([bstr.decode(encoding) for bstr in ncrow]).strip() for ncrow in nclist]

    def _import_2d_observations(self) -> None:
        self.set_logger_message("Reading 2D data")
        for nkey, dkey, map_key, fname in zip(
            [self._key_2d_q_name, self._key_2d_h_name],
            [self._key_2d_q, self._key_2d_h],
            ["2D_Q", "2D_H"],
            [self.file_2d_q, self.file_2d_h],
        ):
            with Dataset(self._path_flow2d) as f:
                self.set_logger_message(f"loading 2D data for {map_key}")
                station_map = pd.read_csv(self.file_1d2d_map, index_col=0)
                qnames = self._parse_names(f.variables[nkey][:])
                qdata = f.variables[dkey][:]

                time = self._parse_time(f.variables["time"])
                station_map_df = pd.DataFrame(columns=station_map.index, index=time)
                self.set_logger_message("Matching 1D and 2D data")
                for _, station in tqdm.tqdm(station_map.iterrows(), total=len(station_map.index)):
                    # Get index of the current station, or skip if ValueError
                    try:
                        si = qnames.index(station[map_key])
                    except ValueError:
                        continue

                    station_map_df[station.name] = qdata[:, si]

                station_map_df.to_csv(f"{fname}")

    def _import_1d_observations(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Import 1D observations.

        time_offset: offset in seconds.
        """
        _file_his = self.path_flow1d

        with Dataset(_file_his) as f:
            names = self._parse_names(f.variables[self._key_1d_h_name])  # names are the same for Q in 1D

            time = self._parse_time(f.variables[self._key_1d_time])
            data = f.variables[self._key_1d_h][:]
            df_h = pd.DataFrame(columns=names, index=time, data=data)

            data = f.variables[self._key_1d_q][:]
            df_q = pd.DataFrame(columns=names, index=time, data=data)

            # apply index shift
            df_h.index = df_h.index + timedelta(seconds=self.time_offset_1d)
            df_q.index = df_q.index + timedelta(seconds=self.time_offset_1d)

            return df_h, df_q

    def _parse_time(self, timevector: pd.DataFrame) -> list[datetime]:
        """Parse time from seconds."""
        unit = timevector.units.replace("seconds since ", "").strip()

        try:
            start_time = datetime.strptime(unit, self._time_fmt)  # noqa: DTZ007
        except ValueError as e:
            if len(e.args) > 0 and e.args[0].startswith("unconverted data remains: "):
                unit = unit[: -(len(e.args[0]) - 26)]
                start_time = datetime.strptime(unit, self._time_fmt)  # noqa: DTZ007

        return [start_time + timedelta(seconds=i) for i in timevector[:]]

    def _parse_1d_stations(self) -> list[str]:
        """Read the names of observations stations from 1D model."""
        return list(self._data_1d_h.columns)

    def _get_1d2d_map(self) -> None:
        _file_his = self.path_flow2d

        with Dataset(_file_his) as f:
            qnames = self._parse_names(f.variables[self._key_2d_q_name][:])
            hnames = self._parse_names(f.variables[self._key_2d_h_name][:])

            # get matching names based on first nine characters
            with self.file_1d2d_map.open("w") as fw:
                fw.write("1D,2D_H,2D_Q\n")
                for n in tqdm.tqdm(list(self._parse_1d_stations())):
                    try:
                        qn = next(x for x in qnames if x.startswith(n[:9]))
                    except StopIteration:
                        qn = ""
                    try:
                        hn = next(x for x in hnames if x.startswith(n[:9]))
                    except StopIteration:
                        hn = ""
                    fw.write(f"{n},{hn},{qn}\n")
