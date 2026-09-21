import cdsapi
import sys


stn_lat = 81.6000
stn_lon = -16.6700

c = cdsapi.Client()

dates = [
    '2024-04-13',
    '2024-04-15',
    '2024-04-17',
    '2024-04-23',
]

for date in dates:

    c.retrieve('reanalysis-era5-complete', {
        'date':    date,
        'levtype': 'ml',
        'levelist': '80/to/137', 
        'param': '54', #surface pressure
        'stream':  'oper',
        'time':    '00/to/23/by/1',
        'type':    'an',
        'area':    f'85/70/-40/20',
        'grid':    '0.25/0.25',
        'format':  'netcdf',
    }, f'data/reanalysis/era5/ERA5_model_levels/era5_stn_ml_{date}.nc')

    print(f"Downloaded data for {date}")



# c.retrieve('reanalysis-era5-single-levels', {
#     'class':   'ea',
#     'date':    dates[0],
#     'variable': '2m_temperature', #surface pressure
#     'stream':  'oper',
#     'time':    '00/to/23/by/1',
#     'area':    f'{stn_lat+.1}/{stn_lon-.1}/{stn_lat-.1}/{stn_lon+.1}',
#     'grid':    '0.25/0.25',
#     'format':  'netcdf',
# }, 'data/era5_stn.netcdf')