import astropy.units as u
from sunpy.net import Fido, attrs as a
from datetime import datetime as dt
import ConfigWrapper
from pathlib import Path
import threading
import sys
import time

"""
Downloads data from VSO using sunpy 0.9.6.

Implements own progress bar because the default one does not appear to work on SCW.

:Author: Richard Grimes
:Version: 1.1.0
:Changelog:
    - V1.1.0 - 2022-02-22: Downloading now tries 3 times to get files.
    - V1.0.1 - 2022-02-12: Now obtains all vso parameters from the config file.
    - V1.0.0 - 2022-02-11: Initial release.
"""


def search(time, instr, phys_obs, cadence):
    """Searches solar data using VSO parameters, returns the result query"""
    log("Searching for data matching the following parameters: "
        "\nTime: {0} to {1}\nInstrument: {2}\nPhysObs: {3}\nCadence: {4}s".format(
        time.start, time.end, instr, phys_obs, cadence))
    result = Fido.search(time, instr, phys_obs, cadence)
    return result


def download(search_result, path):
    """Starts downloading data. Blocking function."""
    return Fido.fetch(search_result, path=str(path), progress=False)


def log(message):
    """Appends a timestamp to log messages"""
    print("[{0}] {1}".format(dt.now().strftime('%H:%M'), message))


# FIXME: Apparently in sunpy 0.9.6 Fido.detch does not return a UnifiedResponse, so this approach is fundamentally flawed.
def brute_force_download(search_result, path, retries=3):
    """
    Repeats the download command multiple times to get as many files as possible.
    Args:
        search_result:
        path:
        retries:

    Returns:

    """
    attempts = 0
    while attempts < retries:
        print("Download attempt {0} of {1}...".format(attempts+1, retries))
        search_result = download(search_result, path)
        attempts += 1
        # if isinstance(search_result,type(list)):
        #     print("Results are a list.")
        #     break
        # if search_result.file_num == 0:
        #     print("No files left over!")
        #     break
    return search_result


# TODO: Maybe try saving the data for resuming later?
def extract_unified_response_date(unified_response):
    """
    Extracts the date from each entry in a unified response so that it can be saved to the disk with json.

    I don't understand how the unified response object is structured and it can't be saved to disk, so I'm
    converting each row of the table to a string and extracting the relevent substring that contains the date.
    I don't like having to do this, and I don't like that it works.
    Args:
        unified_response:

    Returns: list<str>
    """
    json_list = []
    for i in range(unified_response.file_num):
        date_string = str(unified_response[:1,i])[412:431]
        json_list.append(date_string)
    return json_list



if __name__ == "__main__":
    # Load the config file
    config_parser = ConfigWrapper.ConfigWrapper()
    if len(sys.argv) == 0:
        config_parser.read('config.ini')
    else:
        config_parser.read(sys.argv[1])

    # Get variables
    fits_dir = Path(config_parser.get("Directories", "ext_fits"))
    time_start = dt.strptime(config_parser.get('Options', 'start_date'), '%Y-%m-%d_%H-%M-%S')
    time_end = dt.strptime(config_parser.get('Options', 'end_date'), '%Y-%m-%d_%H-%M-%S')
    date = a.vso.Time(time_start, time_end)
    instr = a.vso.Instrument(config_parser.get("Download", "instrument"))
    phys_obs = a.vso.Physobs(config_parser.get("Download", "phys_obs"))
    cadence = a.vso.Sample(config_parser.getint("Download", "cadence") * u.second)
    delay = 60  # Time in seconds to wait between progress updates. Default: 300
    start_fits_file_count = len([x for x in fits_dir.glob('*.fits') if x.is_file()])

    # Search for data
    search_result = search(date, instr, phys_obs, cadence)
    dl_count = search_result.file_num
    log("Found {0} results.".format(dl_count))

    # Download search result in separate thread so we can monitor the progress separately.
    # The progress bar in Fido is only updated at the end or not at all, so separate tracking is needed for
    # an accurate readout.
    dl_thread = threading.Thread(target=brute_force_download, args=(search_result, fits_dir))
    dl_thread.start()

    # Manually check how many files exist on disk and compare to search results for eta.
    finished = False
    last_log_time = None
    while not finished:
        if not dl_thread.is_alive():
            log("Download finished!")
            finished = True
            break
        # Only print the log every <delay> seconds, but keep alive main thread for update checking.
        if last_log_time is None or (dt.now() - last_log_time).total_seconds() > delay:
            total_files = len([x for x in fits_dir.glob('*.fits') if x.is_file()])
            new_files = total_files - start_fits_file_count
            complete_percent = round(new_files / dl_count * 100.0)
            log("{0}/{1} new files downloaded - {2}% complete...".format(new_files, dl_count, complete_percent))
            time.sleep(delay)


    # Final output.
    total_fits = len([x for x in fits_dir.glob('*.fits') if x.is_file()])
    log("There are a total of {0} files in the download directory ({1}).".format(total_fits, str(fits_dir)))

