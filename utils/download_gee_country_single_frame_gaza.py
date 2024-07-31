
import argparse
import requests
import time
import ee
import os


try:
    ee.Initialize()
except:
    print("couldn't init EE")
    ee.Authenticate(auth_mode="localhost")
    ee.Initialize()
    # gcloud auth application-default login --no-browser


"""
This script is used to download single frames of Sentinel 1 and Sentinel 2 data for the given bounding box.
We recommend using ESA's EO browser (https://apps.sentinel-hub.com/eo-browser/) to check for the availability of the data.
The default values in this script are to download single frames in Gaza for the year 2023/24.

"""

ee_crs = ee.Projection('EPSG:4326')

# {
#     'frame4': (f'2023-11-01'),
#     'frame6': (f'2023-11-16'),
#     'frame7': (f'2023-11-21'),
#     'frame8': (f'2023-11-26'),
#     'frame10': (f'2023-12-06'),
#     'frame11': (f'2023-12-11'),
#     'frame14': (f'2023-12-26'),
#     'frame15': (f'2023-12-31'),
#     'frame16': (f'2024-01-10'),
#     'frame18': (f'2024-01-20'),
# }


def get_sentinel2_config():
    return {
        'frame0': (f'2023-09-21', f'2023-09-23'),
        'frame1': (f'2023-09-26', f'2023-09-28'),
        'frame2': (f'2023-10-06', f'2023-10-08'),
        'frame3': (f'2023-10-11', f'2023-10-13'),
        'frame4': (f'2023-10-31', f'2023-11-02'),
        'frame5': (f'2023-11-10', f'2023-11-12'),
        'frame6': (f'2023-11-15', f'2023-11-17'),
        'frame7': (f'2023-11-20', f'2023-11-22'),
        'frame8': (f'2023-11-25', f'2023-11-27'),
        'frame9': (f'2023-11-30', f'2023-12-02'),
        'frame10': (f'2023-12-05', f'2023-12-07'),
        'frame11': (f'2023-12-10', f'2023-12-12'),
        'frame13': (f'2023-12-15', f'2023-12-17'),
        'frame14': (f'2023-12-25', f'2023-12-27'),
        'frame15': (f'2023-12-30', f'2024-01-01'),
        'frame16': (f'2024-01-09', f'2024-01-11'),
        'frame17': (f'2024-01-14', f'2024-01-16'),
        'frame18': (f'2024-01-19', f'2024-01-21'),
        'frame19': (f'2024-01-24', f'2024-01-26'),
    }


def get_sentinel1desc_config():
    return {
        'frame0': (f'2023-09-18', f'2023-09-20'),
        'frame1': (f'2023-09-30', f'2023-10-02'),
        'frame2': (f'2023-10-12', f'2023-10-14'),
        'frame3': (f'2023-10-24', f'2023-10-26'),
        'frame4': (f'2023-11-05', f'2023-11-07'),
        'frame5': (f'2023-11-17', f'2023-11-19'),
        'frame6': (f'2023-11-29', f'2023-12-01'),
        'frame7': (f'2023-12-11', f'2023-12-13'),
        'frame8': (f'2023-12-23', f'2023-12-25'),
        'frame9': (f'2024-01-04', f'2024-01-06'),
        'frame10': (f'2024-01-16', f'2024-01-18'), 
    }

def get_sentinel1asc_config():
    return {
        'frame0': (f'2023-09-17', f'2023-09-19'),
        'frame1': (f'2023-09-22', f'2023-09-24'),
        'frame2': (f'2023-09-29', f'2023-10-01'),
        'frame3': (f'2023-10-04', f'2023-10-06'),
        'frame4': (f'2023-10-11', f'2023-10-13'),
        'frame5': (f'2023-10-16', f'2023-10-18'),
        'frame6': (f'2023-10-23', f'2023-10-25'),
        'frame7': (f'2023-10-28', f'2023-10-30'),
        'frame8': (f'2023-11-04', f'2023-11-06'),
        'frame9': (f'2023-11-16', f'2023-11-18'),
        'frame10': (f'2023-11-21', f'2023-11-23'),
        'frame11': (f'2023-11-28', f'2023-11-30'), 
        'frame12': (f'2023-12-03', f'2023-12-05'), 
        'frame13': (f'2023-12-10', f'2023-12-12'), 
        'frame14': (f'2023-12-15', f'2023-12-17'), 
        'frame15': (f'2023-12-22', f'2023-12-24'), 
        'frame16': (f'2023-12-27', f'2023-12-29'), 
        'frame17': (f'2024-01-03', f'2024-01-05'), 
        'frame18': (f'2024-01-08', f'2024-01-10'), 
        'frame19': (f'2024-01-15', f'2024-01-17'), 
        'frame20': (f'2024-01-20', f'2024-01-22'), 
    }


# 2020 is the default year 

configS2 = get_sentinel2_config()
configS1desc = get_sentinel1desc_config()
configS1asc = get_sentinel1asc_config()

CLOUD_FILTER = 60
CLD_PRB_THRESH = 60
NIR_DRK_THRESH = 0.15
CLD_PRJ_DIST = 2
BUFFER = 60

def start(task):
        # Start the task.
    try:
        task.start()
    except ee.ee_exception.EEException:
        for i in range(128):
            print("Congrats. too-many jobs. EE is at it's limit. Trial", i,". Taking a 15s pause...")
            time.sleep(15)
            try:
                task.start()
            except:
                pass
            else:
                break 
            if i>30:
                raise Exception("Could not submit EE job")
            

def download_tile(url, filename, folder):
    response = requests.get(url)
    if response.status_code == 200:
        with open(os.path.join(folder, filename), 'wb') as f:
            f.write(response.content)
    else:
        print(f"Error downloading {filename}: {response.status_code}")


# https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless
def get_s2_sr_cld_col(aoi, start_date, end_date):
    """Join Sentinel-2 Surface Reflectance and Cloud Probability
    This function retrieves and joins ee.ImageCollections:
    'COPERNICUS/S2_SR' and 'COPERNICUS/S2_CLOUD_PROBABILITY'
    Parameters
    ----------
    aoi : ee.Geometry or ee.FeatureCollection
      Area of interested used to filter Sentinel imagery
    params : dict
      Dictionary used to select and filter Sentinel images. Must contain
      START_DATE : str (YYYY-MM-DD)
      END_DATE : str (YYYY-MM-DD)
      CLOUD_FILTER : int
        Threshold percentage for filtering Sentinel images
    """
    
    # Real Data from raw S2 collection
    s2_sr_col = (ee.ImageCollection('COPERNICUS/S2') 
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER)))
    
    # Query the Sentinel-2 SR collection with cloud masks.
    s2_sr_col_FORMASKS = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') 
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER))
        .select('SCL'))
        
    # Import and filter s2cloudless.
    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        .filterBounds(aoi)
        .filterDate(start_date, end_date))
    # print(s2_cloudless_col.getInfo()["bands"])

    # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
    # return s2_sr_col
    merge1 =  ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary': s2_sr_col,
        'secondary': s2_cloudless_col,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'
        })
    }))


    merge1 = ee.ImageCollection.combine(merge1, s2_sr_col_FORMASKS)

    return merge1


# https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless
def add_cloud_bands(img):
    """Add cloud bands to Sentinel-2 image
    Parameters
    ----------
    img : ee.Image
      Sentinel 2 image including (cloud) 'probability' band
    params : dict
      Parameter dictionary including
      CLD_PRB_THRESH : int
        Threshold percentage to identify cloudy pixels
    """
    # Get s2cloudless image, subset the probability band.
    cld_prb = ee.Image(img.get('s2cloudless')).select('probability')

    # Condition s2cloudless by the probability threshold value.
    is_cloud = cld_prb.gt(CLD_PRB_THRESH).rename('clouds')

    # Add the cloud probability layer and cloud mask as image bands.
    return img.addBands(ee.Image([cld_prb, is_cloud]))


# https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless
def add_shadow_bands(img):
    """Add cloud shadow bands to Sentinel-2 image
    Parameters
    ----------
    img : ee.Image
      Sentinel 2 image including (cloud) 'probability' band
    params : dict
      Parameter dictionary including
      NIR_DRK_THRESH : int
        Threshold percentage to identify potential shadow pixels as dark pixels from NIR band
      CLD_PRJ_DIST : int
        Distance to project clouds along azimuth angle to detect potential cloud shadows
    """
    
    # Identify water pixels from the SCL band.
    not_water = img.select('SCL').neq(6)

    # Identify dark NIR pixels that are not water (potential cloud shadow pixels).
    SR_BAND_SCALE = 1e4
    dark_pixels = img.select('B8').lt(NIR_DRK_THRESH*SR_BAND_SCALE).multiply(not_water).rename('dark_pixels')

    # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
    shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')))

    # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
    cld_proj = (img.select('clouds').directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST*10)
        .reproject(**{'crs': img.select(0).projection(), 'scale': 100})
        .select('distance')
        .mask()
        .rename('cloud_transform'))

    # Identify the intersection of dark pixels with cloud shadow projection.
    shadows = cld_proj.multiply(dark_pixels).rename('shadows')

    # Add dark pixels, cloud projection, and identified shadows as image bands.
    return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))


# https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless
def add_cld_shdw_mask(img):
    # Add cloud component bands.
    img_cloud = add_cloud_bands(img)

    # Add cloud shadow component bands.
    img_cloud_shadow = add_shadow_bands(img_cloud)

    # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
    is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)

    # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
    # 20 m scale is for speed, and assumes clouds don't require 10 m precision.
    is_cld_shdw = (is_cld_shdw.focalMin(2).focalMax(BUFFER*2/20)
        .reproject(**{'crs': img.select([0]).projection(), 'scale': 20})
        .rename('cloudmask'))

    # Add the final cloud-shadow mask to the image.
    # return img.addBands(is_cld_shdw)
    # return img_cloud_shadow.addBands(is_cld_shdw)
    return img.addBands(is_cld_shdw)


# https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless
def apply_cld_shdw_mask(img):
    # Subset the cloudmask band and invert it so clouds/shadow are 0, else 1.
    not_cld_shdw = img.select('cloudmask').Not()

    # Subset reflectance bands and update their masks, return the result.
    return img.select('B.*').updateMask(not_cld_shdw)


def submit_s2job(s2_sr_mosaic, description, name, exportarea):
    # Submits a job to Google earth engine with all the requred arguments
    
    task = ee.batch.Export.image.toDrive(
        image=s2_sr_mosaic,
        scale=10,  
        description=description + "_" + name,
        fileFormat="GEOTIFF", 
        folder=name, 
        region=exportarea,
        crs='EPSG:4326', #OLD
        # crs='EPSG:3035', #NEW, but wrong, this only works for Europe
        maxPixels=80000000000 
    )

    # submit/start the job
    task.start() 


# New function to download the Sentine2 data
def export_cloud_free_sen2(season, dates, roi_id, roi, debug=0, S2type="S2"):
    """
    Export cloud free Sentinel-2 data for a given season and region of interest
    Parameters
    ----------
    season : str
        Season to download data for
    dates : list
        List of dates to download data for
    roi_id : str
        Region of interest ID
    roi : ee.Geometry
        Region of interest
    debug : int
        Debug level
    -------
    Returns
        None
    """

    if S2type == "S2":
        bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12']
    elif S2type == "S2_SR_HARMONIZED":
        bands = ['B2', 'B3', 'B4', 'B8']
    
    # Get the start and end dates for the season.
    start_date = ee.Date(dates[0])
    end_date = ee.Date(dates[1])

    # Get the Sentinel-2 surface reflectance and cloud probability collections.
    s2_sr = ee.ImageCollection("COPERNICUS/" + S2type)
    s2_clouds = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")

    # Filter the collections by the ROI and date.
    criteria = ee.Filter.And(ee.Filter.bounds(roi), ee.Filter.date(start_date, end_date))

    # Filter the collections by the ROI and date.
    s2_sr = s2_sr.filter(criteria)
    s2_clouds = s2_clouds.filter(criteria)

    # Join S2 SR with cloud probability dataset to add cloud mask.
    join = ee.Join.saveFirst('cloud_mask')
    condition = ee.Filter.equals(leftField='system:index', rightField='system:index')
    s2_sr_with_cloud_mask = join.apply(primary=s2_sr, secondary=s2_clouds, condition=condition)

    # Define a function to mask clouds using the probability threshold.
    def mask_clouds(img):
        clouds = ee.Image(img.get('cloud_mask')).select('probability')
        is_not_cloud = clouds.lt(65)
        return img.updateMask(is_not_cloud)

    # Map the function over one year of data and take the median.
    img_c = ee.ImageCollection(s2_sr_with_cloud_mask).map(mask_clouds)

    # Get the median of each pixel for the time period.
    cloud_free = img_c.median()
    # filename = f"{roi_id}_{season}"
    filename = f"{season}_{roi_id}"

    
    # Export the image to Google Drive.
    task = ee.batch.Export.image.toDrive(
        image=cloud_free.select(bands),
        description=filename,
        scale=10,
        region=roi,
        folder=f"{roi_id}",
        fileNamePrefix=filename,
        maxPixels=1e13
    )

    task.start()

def export_S1_tile(season, dates, filename, roi, folder, scale=10, crs='EPSG:4326', url_mode=True, orbit='DESCENDING'):
    """
    Export Sentinel-1 data for a given season and region of interest
    """
    start_date = ee.Date(dates[0])
    end_date = ee.Date(dates[1])

    # Define a method for filtering and compositing.
    collectionS1 = ee.ImageCollection('COPERNICUS/S1_GRD')
    collectionS1 = collectionS1.filter(ee.Filter.eq('instrumentMode', 'IW'))
    collectionS1 = collectionS1.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    collectionS1 = collectionS1.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    collectionS1 = collectionS1.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
    collectionS1 = collectionS1.filterBounds(roi)
    # collectionS1 = collectionS1.filter(ee.Filter.contains('.geo', roi))
    collectionS1 = collectionS1.filterDate(start_date, end_date)
    collectionS1 = collectionS1.select(['VV', 'VH'])
    collectionS1_first_desc = collectionS1.median() 


    # also for acending orbit (NOT USED)
    collectionS1 = ee.ImageCollection('COPERNICUS/S1_GRD')
    collectionS1 = collectionS1.filter(ee.Filter.eq('instrumentMode', 'IW'))
    collectionS1 = collectionS1.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    collectionS1 = collectionS1.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    collectionS1 = collectionS1.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
    collectionS1 = collectionS1.filterBounds(roi)
    # collectionS1 = collectionS1.filter(ee.Filter.contains('.geo', roi))
    collectionS1 = collectionS1.filterDate(start_date, end_date)
    collectionS1 = collectionS1.select(['VV', 'VH'])
    collectionS1_first_asc = collectionS1.median() 

    if orbit == 'ASCENDING':
        target_collection = collectionS1_first_asc
    elif orbit == 'DESCENDING':
        target_collection = collectionS1_first_desc


    # NOTE: we only use the descending orbit for consistency
    if url_mode:
        try:
            url = target_collection.getDownloadUrl({
                'scale': scale,
                'format': "GEOTIFF", 
                'region': roi,
                'crs': crs,
                'maxPixels':80000000000,
            })
        except Exception as e:
            print(e)
            print("Error in " + filename + " getting the url, moving on tho the next tile")
            return None
        
        download_tile(url, filename, folder)
        return url

    # Export the image, specifying scale and region.
    task = ee.batch.Export.image.toDrive(
        image = target_collection,
        scale = scale,  
        description = filename, 
        fileFormat="GEOTIFF",  
        folder=folder, 
        region = roi, 
        crs=crs, 
        maxPixels=80000000000,
    )
    start(task)

    return None

# def export_gbuildings(collection_name, confidence_min, bbox, description, folder, scale=10):
def export_gbuildings(roi, filename, folder, confidence_min=0.0, scale=10, crs='EPSG:4326', btype="v3"):
    """
    Function to export a filtered Google Earth Engine collection to Google Drive.

    Args:
    - collection_name (str): name of the GEE collection to filter and export.
    - confidence_min (float): minimum confidence to filter by.
    - roi (list): bounding box to filter by, in the format [minLon, minLat, maxLon, maxLat].
    - filename (str): description for the exported data.
    - folder (str): name of the folder in Google Drive to export the data to.
    - scale (int): resolution of the export in meters (default is 30).
    - crs (str): coordinate reference system of the exported data (default is 'EPSG:4326').

    Returns:
    - None.
    """

    # Load the building footprint dataset
    t = ee.FeatureCollection('GOOGLE/Research/open-buildings/{type}/polygons'.format(type=btype))

    # Apply the confidence filters and clip to the bounding box
    # t_filtered = t.filter(ee.Filter.gte('confidence', confidence_min)).filterBounds(roi)
    t_filtered = t.filterBounds(roi)

    # Define the export parameters
    export_params = {
        'collection': t_filtered,
        'description': filename,
        'folder': folder,
    }

    # Export the data to Google Drive
    task = ee.batch.Export.table.toDrive(**export_params)

    # Start the task
    start(task)


def download(minx, miny, maxx, maxy, name):
    """
    Function to download the data from Google Earth Engine to Drive.
    Inputs:
    - minx, miny, maxx, maxy (float): coordinates of the bounding box.
    - name (str): name of the file to download.
    Returns:
    - None. (the files are downloaded to Drive instead)
    """

    exportarea = { "type": "Polygon",  "coordinates": [[[maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny], [maxx, miny]]]  }
    exportarea = ee.Geometry.Polygon(exportarea["coordinates"]) 

    # transform into local projection
    # find the EPSG code for the local projection
    # exportarea3035 = exportarea.transform('EPSG:3035')

    S1 = True
    S2 = False
    S2A = True
    VIIRS = False
    GoogleBuildings = False 

    if S1:
        ########################### Processing Sentinel 1 #############################################
        
        # ascending
        for season in configS1asc.keys():
            start_date, finish_date = configS1asc[season]
            export_S1_tile(season, (start_date, finish_date), f"S1dasc_{season}_" + name, exportarea, name,
                           url_mode=False, orbit='ASCENDING')

    if S2A:
        ########################### Processing Sentinel 2 Level 2A #############################################

        for season in configS2:
            start_date, finish_date = configS2[season]
            export_cloud_free_sen2(f"S2A_{season}", (start_date, finish_date), name, exportarea, S2type="S2_SR_HARMONIZED")
     

    if GoogleBuildings:
        ########################### Processing Google Buildings #############################################
        
        # Google Buildings
        export_gbuildings(exportarea, "Gbuildings_" + name, folder=name, confidence_min=0.0, btype="v3") 
        export_gbuildings(exportarea, "Gbuildings_v1_" + name, folder=name, confidence_min=0.0, btype="v1")

    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("minx", type=float)
    parser.add_argument("miny", type=float)
    parser.add_argument("maxx", type=float)
    parser.add_argument("maxy", type=float) 
    parser.add_argument("name", type=str) 
    args = parser.parse_args()

    download(args.minx, args.miny, args.maxx, args.maxy, args.name)


if __name__ == "__main__":
    main()
    print("Done!")


