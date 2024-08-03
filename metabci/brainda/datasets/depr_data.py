"""
Alex Motor imagery dataset.
"""

from typing import Union, Optional, Dict, List, cast
from pathlib import Path

from mne.io import Raw, read_raw_eeglab
from mne.channels import make_standard_montage
from .base import BaseDataset
from ..utils.download import mne_data_path
from ..utils.channels import upper_ch_names

#local data path
DREP445_URL = "data"

class DREP445(BaseDataset):
    _EVENTS = {
        "close_eyes": (11, (0, 240)),
    }

    _CHANNELS = [
        'Fp1',
        'Fp2',
        'F3',
        'F4',
        'F7',
        'F8',
        'FC1',
        'FC2',
        'FC5',
        'FC6',
        'C3',
        'C4',
        'CP1',
        'CP2',
        'CP5',
        'CP6',
        'Pz',
        'P3',
        'P4',
        'Oz',
        'O1',
        'O2'
    ]

    def __init__(self):
        super().__init__(
            dataset_code="depr445",
            subjects=list(range(1, 3)),
            events=self._EVENTS,
            channels=self._CHANNELS,
            srate=250,
            paradigm="oddball",
        )

    def data_path(
        self,
        subject: Union[str, int],
        path: Optional[Union[str, Path]] = None,
        force_update: bool = False,
        update_path: Optional[bool] = None,
        proxies: Optional[Dict[str, str]] = None,
        verbose: Optional[Union[bool, str, int]] = None,
    ) -> List[List[Union[str, Path]]]:
        subject = cast(int, subject)
        if subject not in self.subjects:
            raise (ValueError("Invalid subject id"))

        if subject in range(1, 18):
            file_name = "抑郁/D{:02d}.mat".format(subject)
        else:
            file_name = "正常/H{:02d}.mat".format(subject-17)

        url = r"file://C:/metabci/MetaBCI/data/{:s}".format(file_name)
        dests = [
            [
                mne_data_path(
                    url,
                    "drep",
                    path=path,
                    proxies=proxies,
                    force_update=force_update,
                    update_path=update_path,
                )
            ]
        ]
        return dests

    def _get_single_subject_data(
        self, subject: Union[str, int], verbose: Optional[Union[bool, str, int]] = None
    ) -> Dict[str, Dict[str, Raw]]:
        dests = self.data_path(subject)
        montage = make_standard_montage("standard_1005")
        montage.rename_channels(
            {ch_name: ch_name.upper() for ch_name in montage.ch_names}
        )
        # montage.ch_names = [ch_name.upper() for ch_name in montage.ch_names]

        sess = dict()
        for isess, run_dests in enumerate(dests):
            runs = dict()
            for irun, run_file in enumerate(run_dests):
                raw = Raw(run_file, preload=True)
                raw = upper_ch_names(raw)
                raw.set_montage(montage)
                runs["run_{:d}".format(irun)] = raw
            sess["session_{:d}".format(isess)] = runs

        return sess