from pathlib import Path
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as matdates

plt.switch_backend('Agg')
# /scratch/a.oln2/sunspots/NOAA_12565_test/fits
file_path = Path("/scratch/a.oln2/sunspots/NOAA_12565_test/fits")
output_path = Path("/scratch/a.oln2/sunspots/NOAA_12565_test/output/")

files = [x for x in file_path.glob('*.fits') if x.is_file()]
dates = []
y = []

for file in files:
    formats = ['hmi_ic_45s_%Y_%m_%d_%H_%M_%S_tai_continuum.fits',
               'hmi_ic_45s_%Y_%m_%d_%H_%M_%S_tai_continuum.0.fits',
               'hmi.ic_45s.%Y.%m.%d_%H_%M_%S_TAI.continuum.fits',
               'hmi.ic_45s.%Y.%m.%d_%H_%M_%S_TAI.continuum.0.fits']
    for format in formats:
        try:
            date = datetime.datetime.strptime(file.name, format)
            break
        except ValueError:
            pass
        raise ValueError("File did not match any expected format template.")
    dates.append(date)
    y.append(1)

fig = plt.figure(figsize=(16,9), dpi=90)
ax = plt.subplot(111)
ax.plot_date(dates, y, ms=((72. / fig.dpi) ** 2))
ax.xaxis.set_major_formatter(matdates.DateFormatter("%d/%m"))
plt.savefig(output_path / "fits_coverage.jpg")
plt.close()
