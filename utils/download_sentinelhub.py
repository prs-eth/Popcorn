import os
import glob
from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    DownloadRequest,
    MimeType,
    MosaickingOrder,
    SentinelHubDownloadClient,
    SentinelHubRequest,
    SHConfig,
    bbox_to_dimensions,
)
from osgeo import gdal

# set oauth client here: https://apps.sentinel-hub.com/dashboard/#/account/settings 
# (only once)
config = SHConfig(
    sh_client_id = 'REPLACE_WITH_YOUR_OWN', # TODO: REPLACE WITH YOUR OWN!!!
    sh_client_secret = 'REPLACE_WITH_YOUR_OWN', # TODO: REPLACE WITH YOUR OWN!!!
    sh_base_url='https://services.sentinel-hub.com')

def get_S1_dates(roi, subcat):
    dates = {
        'bgd'   :   { #asc
            '09SA': (f'2018-03-20', f'2018-03-22'), 
            '10B': (f'2018-04-13', f'2018-04-15'), 
            '10SA': (f'2018-05-31', f'2018-06-02'), 
            '11B': (f'2018-06-12', f'2018-06-14'), 
            '11SA': (f'2018-07-18', f'2018-07-20'), 
            '12SA': (f'2018-10-10', f'2018-10-12'),
            '13SA': (f'2018-11-27', f'2018-11-29'), 
            '14SA': (f'2019-03-15', f'2019-03-17'),
            '15SA': (f'2019-07-01', f'2019-07-03')
        },
        'eth'   :   { #desc
            'gambela' : (f'2019-10-29', f'2019-11-01'),
            'okugo' : (f'2010-10-30', f'2019-11-01')
        },
        'sdn'   :   { # desc
            'white_nile' : (f'2020-02-19', f'2020-02-21')
        },
        'tcd'   :   { #desc
            'logone_oriental': (f'2021-10-22', f'2021-10-24'),
            'moyen_chari': (f'2021-10-17', f'2021-10-19'),
            'ouaddai' : (f'2021-10-24', f'2021-10-26'),
            'sila' : (f'2021-10-24', f'2021-10-26'),
            'wadi_fira' : (f'2021-11-05', f'2021-11-07')
        },
        'uga'   :   { #desc
            'adjumani' : (f'2020-10-17', f'2020-10-19'),
            'northwest' : (f'2020-10-16', f'2020-10-19'),
            'northeast' : (f'2020-10-17', f'2020-10-19')
        },
    }
    return dates[roi][subcat]

def get_S1_collection(roi):
    if roi == 'bgd':
        collection = DataCollection.SENTINEL1_IW_ASC
    else:
        collection = DataCollection.SENTINEL1_IW_DES
    return collection

def get_S2_dates(roi, subcat):
    # chosen manually to be as close to the data colllection point as possible 
    # (and also cloudless in the areas where the refugee camps are)
    dates = {
        'bgd'   :   {
            '09SA': (f'2018-03-09', f'2018-03-11'),
            '10B': (f'2018-03-29', f'2018-03-31'),
            '10SA': (f'2018-03-29', f'2018-03-31'),
            '11B': (f'2018-09-15', f'2018-09-17'), 
            '11SA': (f'2018-09-15', f'2018-09-17'),
            '12SA': (f'2018-10-15', f'2018-10-17'),
            '13SA': (f'2018-11-14', f'2018-11-16'),
            '14SA': (f'2019-03-19', f'2019-03-21'),
            '15SA': (f'2019-04-23', f'2019-04-25')
        },
        'eth'   :   {
            'gambela' : (f'2019-10-31', f'2019-11-02'),
            'okugo' : (f'2019-10-06', f'2019-10-08')
        },
        'sdn'   :   {
            'white_nile' : (f'2020-02-26', f'2020-02-28')
        },
        'tcd'   :   {
            'logone_oriental' : (f'2021-10-21', f'2021-10-23'),
            'moyen_chari' : (f'2021-10-28', f'2021-10-30'),
            'ouaddai' : (f'2021-10-25', f'2021-10-30'), 
            'sila' : (f'2021-10-30', f'2021-11-01'),
            'wadi_fira' : (f'2021-10-28', f'2021-11-01') 
        },
        'uga'   :   { 
            'adjumani' : (f'2020-12-12', f'2020-12-16'),
            'northwest' : (f'2020-12-12', f'2020-12-14'),
            'northeast' : (f'2020-12-14', f'2020-12-16')
        },
    }
    return dates[roi][subcat]

def get_coords(roi, subcat):
    # bounding boxes of shapefiles
    coords_all = {
        'bgd'   :   (92.08, 20.87, 92.31, 21.27),
        'eth'   :   {
            'gambela' : (34.10, 7.58, 34.80, 8.39),
            'okugo' : (35.07, 6.42, 35.20, 6.55)
        },
        'sdn'   :   (32.70, 12.30, 32.90, 12.80),
        'tcd'   :   {
            'logone_oriental': (16.41, 7.87, 16.70, 8.23),
            'moyen_chari': (18.69, 8.37, 18.82, 8.52),
            'ouaddai' : (21.12, 13.37, 21.86, 13.87), 
            'sila' : (21.23, 11.82, 21.45, 12.29), 
            'wadi_fira' : (21.86, 14.39, 22.41, 15.19)
        },
        'uga'   :   {
            'adjumani' : (31.58, 3.11, 32.10, 3.54),
            'northwest' : (31.10, 2.92, 31.80, 3.64),
            'northeast' : (32.31, 3.20, 32.59, 3.48)
        },
    }
    if roi in ['bgd', 'sdn']:
        coords = coords_all[roi]
    else:
        coords = coords_all[roi][subcat]
    return coords


def split_x(coords):
    y_min, y_max = coords[1], coords[3]
    x_min, x_mid, x_max = coords[0], coords[0] + (coords[2]-coords[0])/2, coords[2]
    coords_west = (x_min, y_min, x_mid, y_max)
    coords_east = (x_mid, y_min, x_max, y_max)
    return [coords_east,coords_west]

def split_y(coords):
    y_min, y_mid, y_max = coords[1], coords[1] + (coords[3]-coords[1])/2, coords[3]
    x_min, x_max = coords[0], coords[2]
    coords_south = (x_min, y_min, x_max, y_mid)
    coords_north = (x_min, y_mid, x_max, y_max)
    return [coords_south, coords_north]

def split_bbox(size, coords):
    
    width, height = size
    coords = [coords]
    
    while height > 2500:
        coords_old = coords
        coords = []
        for c in coords_old:
            coords.extend(split_y(c))
        bbox = BBox(bbox=coords[0], crs=CRS.WGS84)
        height = bbox_to_dimensions(bbox, resolution=10)[1]
        
    while width > 2500:
        coords_old = coords
        coords = []
        for c in coords_old:
            coords.extend(split_x(c))
        bbox = BBox(bbox=coords[0], crs=CRS.WGS84)
        width = bbox_to_dimensions(bbox, resolution=10)[0]
    
    bbs = []    
    for c in coords:
        bbox = BBox(bbox=c, crs=CRS.WGS84)
        bbs.append(bbox)
    
    return bbs

# https://gis.stackexchange.com/questions/449569/merging-a-large-number-of-geotiff-files-via-gdal-merge-py
def merge(datapath,outfile):
    
    infiles = []
    for folder, _, filenames in os.walk(datapath):
        print(folder)
        for filename in filenames:
            if filename[-4:] == 'tiff':
                infiles.append(os.path.join(folder, filename))
    
    #Build VRT from input files 
    vrt_file = "merged.vrt"
    gdal.BuildVRT(vrt_file, infiles)

    # Translate VRT to TIFF
    gdal.Translate(outfile, vrt_file)
    
    os.remove(vrt_file)
    
    return None

def build_request(roi, season, data, bbox, size):
    if data == 'S2A':
        evalscript = """
            //VERSION=3

            function setup() {
                return {
                    input: [{
                        bands: ["B02", "B03", "B04", "B08"],
                        units: ["DN", "DN", "DN", "DN"]
                    }],
                    output: {
                        bands: 4,
                        sampleType: "UINT16"
                    }
                };
            }

            function evaluatePixel(sample) {
                return [sample.B02, sample.B03, sample.B04, sample.B08];
            }
        """
        collection = DataCollection.SENTINEL2_L2A
    
        start_date, end_date = get_S2_dates(roi, season)
        
    if data == 'S1':
        
        evalscript = """
            //VERSION=3

            function setup() {
                return {
                    input: [{
                        bands: ["VV", "VH"]
                    }],
                    output: {
                        bands: 2,
                        sampleType: "FLOAT32"
                    }
                };
            }

            function evaluatePixel(sample) {
                return [sample.VV, sample.VH];
            }
        """
        collection = get_S1_collection(roi)
    
        start_date, end_date = get_S1_dates(roi, season)
    
    request = SentinelHubRequest(
            evalscript=evalscript,
            data_folder = os.path.join("/scratch3/data/ref_camps_data/merged/EE",roi, f"{data}{season}"),
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection= collection,
                    time_interval=(start_date, end_date)
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=bbox,
            size=size,
            config=config,
        )
    
    return request

# reference: https://github.com/sentinel-hub/sentinelhub-py/tree/master/examples
def download(coords, roi, season, data):
    
    outfile = os.path.join("/scratch3/data/ref_camps_data/merged/EE",roi, f"{data}{season}", f"{roi}_{data}{season}.tiff")
    
    resolution = 10 # in meter
    bbox = BBox(bbox=coords, crs=CRS.WGS84)
    size = bbox_to_dimensions(bbox, resolution=resolution)
    
    bboxes = [bbox]
    split = False

    # Max image size that can be downloaded is 2500 x 2500
    if size[0] > 2500 or size[1] > 2500:
        bboxes = split_bbox(size, coords)
        split = True

    
    for bbox in bboxes:
        size = bbox_to_dimensions(bbox, resolution=resolution)
        request = build_request(roi, season, data, bbox, size)

        request.save_data()
    
    if split:
        merge(request.data_folder, outfile)
        
    else:
        for folder, _, filenames in os.walk(request.data_folder):
            for filename in filenames:
                if filename[-4:] == 'tiff':
                    print(os.path.join(folder, filename))
                    os.rename(os.path.join(folder, filename), outfile)
        
    return None


def main():
    
    roi = 'uga'
    data = ['S1', 'S2A']

    if roi == 'bgd':
        seasons = ['09SA', '10B', '10SA', '11B', '11SA', '12SA', '13SA', '14SA', '15SA']
        coords = get_coords(roi, seasons)
        for season in seasons:
            download(coords, roi, season, data[0])
            download(coords, roi, season, data[1])
    
    if roi == 'eth':
        regions = ['gambela', 'okugo']
        for region in regions:
            coords = get_coords(roi, region)
            download(coords, roi, region, data[0])
            download(coords, roi, region, data[1])
            
    if roi == 'sdn':
        coords = get_coords(roi, 'white_nile')
        download(coords, roi, 'white_nile', data[0])
        download(coords, roi, 'white_nile', data[1])
        
    if roi == 'tcd':
        regions = ['logone_oriental', 'moyen_chari', 'ouaddai', 'sila', 'wadi_fira']
        for region in regions:
            coords = get_coords(roi, region)
            download(coords, roi, region, data[0])
            download(coords, roi, region, data[1])
            
    if roi == 'uga':
        regions = ['adjumani', 'northeast', 'northwest']
        for region in regions:
            coords = get_coords(roi, region)
            download(coords, roi, region, data[0])
            download(coords, roi, region, data[1])


if __name__ == "__main__":
    main()
    print("Done!")
