import os
import requests
from astropy.io import fits
import numpy as np
import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class SRHCorrPlot:
    base_url = "https://ftp.rao.istp.ac.ru/SRH/corrPlot/"
    array_map = dict(SRH0306="0306", SRH0612="0612", SRH1224="1224")

    def __init__(
        self, this_date, array, frequency, cache_dir="datasets/corrplot_cache"
    ):
        self.dt = this_date
        self.array = array
        self.frequency = frequency
        self.date_url = self.base_url + this_date.strftime(
            f"%Y/%m/srh_{self.array_map[array]}_cp_%Y%m%d.fits"
        )
        self.cache_dir = cache_dir
        self.fits_file = os.path.join(
            self.cache_dir,
            f"srh_{self.array_map[array]}_cp_{this_date.strftime('%Y%m%d')}.fits",
        )
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        self.data = None
        self.download_and_open_fits()

    def download_and_open_fits(self):
        if not os.path.exists(self.fits_file):
            response = requests.get(self.date_url)
            if response.status_code == 200:
                with open(self.fits_file, "wb") as f:
                    f.write(response.content)
                print(f"Downloaded FITS file to {self.fits_file}")
            else:
                print(f"Failed to download file: {response.status_code}")
                return None, None

        try:
            with fits.open(self.fits_file) as hdul:
                self.frequencies = np.array([int(x[0] / 1e3) for x in hdul[1].data])
                self.data = hdul[2].data

                if self.frequency not in self.frequencies:
                    self.frequency = self.frequencies[0]
                self.frequency_index = np.where(self.frequencies == self.frequency)[0][
                    0
                ]
                corr_data = self.data[self.frequency_index]
                times = corr_data.field("time")
                self.I = corr_data.field("I")
                self.V = corr_data.field("V")
                self.flux_I = corr_data.field("flux_I")
                self.flux_V = corr_data.field("flux_V")
                field_names = self.data.dtype.names
                if "flux_RCP" in field_names and "flux_LCP" in field_names:
                    self.flux_RCP = corr_data.field("flux_RCP")
                    self.flux_LCP = corr_data.field("flux_LCP")
                else:
                    self.flux_RCP = None
                    self.flux_LCP = None
                self.times = [
                    datetime.datetime(self.dt.year, self.dt.month, self.dt.day, 0, 0, 0)
                    + datetime.timedelta(seconds=s)
                    for s in times
                ]

        except Exception as e:
            print(f"Error opening FITS file: {e}")
            return None, None

    def overplot_to(self, ax, dataset_name="I"):
        if self.data is None:
            print("No data available for plotting.")
            return

        dataset_to_plot = getattr(self, dataset_name)
        ax.plot(self.times, dataset_to_plot, label=dataset_name)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
        ax.set_title(f"{self.frequency} MHz, {self.dt}")
        ax.grid(True)
        ax.legend()
        ax.set_xlim(self.times[0], self.times[-1])
        ax.set_xlabel("Time, UT")
        ax.set_ylabel("Value, a.u.")
