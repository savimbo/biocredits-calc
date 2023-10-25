import os
import requests
import shutil
import json
import time
import subprocess
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
import math
import numpy as np
import pandas as pd
import plotly.express as px
from shapely import wkt
import contextily as ctx
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def load_config():
    with open('config.json', 'r') as f:
        return json.load(f)

def download_kml_official(save_directory='KML/'):
    """
    Download KML files from Airtable
    """
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    else:
        shutil.rmtree(save_directory)
        os.makedirs(save_directory)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    config = load_config()
    BASE_ID = config['KML_TABLE']['BASE_ID']
    TABLE_NAME = config['KML_TABLE']['TABLE_NAME']
    FIELD = config['KML_TABLE']['FIELD']
    PERSONAL_ACCESS_TOKEN = config['PERSONAL_ACCESS_TOKEN']

    SAVE_DIRECTORY = save_directory
    AIRTABLE_ENDPOINT = f"https://api.airtable.com/v0/{BASE_ID}/{TABLE_NAME}"
    headers = {
        "Authorization": f"Bearer {PERSONAL_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }

    response = requests.get(AIRTABLE_ENDPOINT, headers=headers)
    response_json = response.json()

    for record in response_json['records']:
        field = record['fields'].get(FIELD)
        if field:
            url = field[0]['url']
            filename = field[0]['filename']
            save_path = os.path.join(SAVE_DIRECTORY, filename)
            
            with requests.get(url, stream=True) as file_response:
                with open(save_path, 'wb') as file:
                    for chunk in file_response.iter_content(chunk_size=8192):
                        file.write(chunk)
            print(f"Downloaded {filename}")

def kml_to_shp(source_directory='KLM/', destination_directory='SHP/'):
    # Ensure the destination directory exists
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
    else:
        shutil.rmtree(destination_directory)
        os.makedirs(destination_directory)
    # Loop through all .kml files in the source directory
    for filename in os.listdir(source_directory):
        if filename.endswith('.kml'):
            base_name = os.path.splitext(filename)[0]  # Get the file name without extension
            source_file_path = os.path.join(source_directory, filename)
            destination_file_path = os.path.join(destination_directory, base_name + '.shp')
            # Convert KML to SHP using ogr2ogr
            cmd = ['ogr2ogr', '-f', 'ESRI Shapefile', destination_file_path, source_file_path]
            subprocess.run(cmd)
            print(f"Converted {filename} to {base_name}.shp")

def load_shp(directory='SHP/'):
    # Loop through all .shp files in the directory and load them
    gdfs = {}
    for filename in os.listdir(directory):
        if filename.endswith('.shp'):
            filepath = os.path.join(directory, filename)
            gdf = gpd.read_file(filepath)
            base_name = os.path.splitext(filename)[0]
            gdfs[base_name] = gdf
    return gdfs

def reorder_polygon(polygon):
    # Extract the x and y coordinates of the polygon
    x, y = polygon.exterior.coords.xy
    # Compute the centroid of the polygon
    centroid = (sum(x) / len(x), sum(y) / len(y))
    # Function to compute the polar angle from the centroid
    def compute_polar_angle(point):
        return math.atan2(point[1] - centroid[1], point[0] - centroid[0])
    # Sort the points by their polar angle
    sorted_points = sorted(polygon.exterior.coords, key=compute_polar_angle)
    # Create a new polygon from the sorted points
    return Polygon(sorted_points)

def set_z_to_zero(coord):
    """Given a coordinate tuple, set its Z value to zero."""
    x, y, _ = coord
    return (x, y, 0)

def normalize_shps(gdfs, reorder_lands=[]):
    """
    Given a list with .shp files loaded as GeoDataFrames, then:
        1) ensure the third coordinates are zero
        2) convert them to polygons
        3) reorder the vertices in clockwise order (if included in reorder_fincas)
        4) return a GeoDataFrame with the polygons 
    """
    lands = {}
    point_shps = []
    polygon_shps = []
    for key in gdfs.keys():
        geometries = gdfs[key]['geometry']
        if len(geometries) == 0:
            print(f"warning: {key} GeoDataFrame is empty!")
            continue

        # Convert the series of points to a list of coordinate tuples with Z set to zero
        if isinstance(geometries.iloc[0], Point): 
            point_shps.append(key)     
            coords = [set_z_to_zero(point.coords[0]) for point in geometries]
        elif isinstance(geometries.iloc[0], Polygon):
            polygon_shps.append(key)
            coords = [set_z_to_zero(coord) for coord in geometries.iloc[0].exterior.coords]
        else:
            raise ValueError("Unsupported geometry type!")
        lands[key] = Polygon(coords)
        
    insert_log_entry('Point KMLs:', str(point_shps))
    insert_log_entry('Polygon KMLs:', str(polygon_shps))
    return lands

def reorder_polygons(gdfs, reorder_lands=[]):
    lands = {}
    for key in gdfs.keys():
        polygon = gdfs[key]
        coords = polygon.exterior.coords
        # Reorder the vertices in clockwise order if needed
        if key in reorder_lands:
            polygon = reorder_polygon(Polygon(coords))
        else:
            polygon = Polygon(coords)
        lands[key] = polygon
    return lands

def shp_to_land(lands, crs = "EPSG:4326"):
    lands = gpd.GeoSeries(lands)
    lands = gpd.GeoDataFrame(lands, columns=['geometry'])
    lands.crs = crs
    return lands

def plot_land(lands, filename='lands.html'):
    # plot fincas with plotly
    centroid = lands[lands['geometry'].is_valid].unary_union.centroid
    fig = px.choropleth_mapbox(lands, geojson=lands.geometry, locations=lands.index, color=lands.index,
                                color_discrete_sequence=["red"], zoom=9.8, center = {"lat": centroid.coords.xy[1][0], "lon": centroid.coords.xy[0][0]},
                                opacity=0.5, labels={'index':'Finca'})
    fig.update_layout(mapbox_style="carto-positron")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    # hide the legend
    fig.update_layout(showlegend=False)
    # make square
    fig.update_layout(height=600, width=600)
    # save 
    fig.write_html(filename)
    #fig.show()

def download_observations():
    """
    Download KML files from Airtable
    """
    config = load_config()
    BASE_ID = config['OBS_TABLE']['BASE_ID']
    TABLE_NAME = config['OBS_TABLE']['TABLE_NAME']
    PERSONAL_ACCESS_TOKEN = config['PERSONAL_ACCESS_TOKEN']
    AIRTABLE_ENDPOINT = f"https://api.airtable.com/v0/{BASE_ID}/{TABLE_NAME}"

    headers = {
        "Authorization": f"Bearer {PERSONAL_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }

    all_records = []
    offset = None

    while True:
        params = {}
        if offset:
            params["offset"] = offset

        response = requests.get(AIRTABLE_ENDPOINT, headers=headers, params=params)
        response_json = response.json()
        
        records = response_json.get('records')
        all_records.extend(records)
        
        offset = response_json.get('offset')
        if not offset:
            break

    insert_log_entry('Total observations fetched:', str(len(all_records)))

    records = pd.DataFrame([r['fields'] for r in all_records])
    # keep records with NIVEL de CREDITO (from SPECIES (ES))
    records = records[[not r is np.nan for r in records["NIVEL de CREDITO (from SPECIES (ES))"]]]
    insert_log_entry('Observations with NIVEL de CREDITO (from SPECIES (ES)):', str(len(records)))
    # transform and create columns
    records['species_id'] = records['species_id'].apply(lambda x: x[0])
    records['name_common'] = records['SPECIES ID (EN) (from SPECIES (ES))'].apply(lambda x: x[0] if type(x) == list else x)
    records['name_latin'] = records['LATIN NAME/NOMBRE LATINO (from SPECIES (ES))'].apply(lambda x: x[0] if type(x) == list else x)
    records['score'] = records['NIVEL de CREDITO (from SPECIES (ES))'].apply(lambda x: x[0])
    records['radius'] = records['species_radius'].apply(lambda x: max(x))   # using max radius of row
    # We are assuming one observation per record, so we can use the first element of the list, max radius, etc.

    # filter records with radius > 0 and eco_long < 0
    records = records.query('radius>0')
    insert_log_entry('Observations with radius > 0:', str(len(records)))
    records = records.query('eco_long<0')
    insert_log_entry('Observations with eco_long < 0:', str(len(records)))

    # renaming and keeping columns
    records = records.rename(columns={'# ECO':'eco_id', 'eco_lat':'lat', 'eco_long':'long'})
    keep_columns = ['eco_id','eco_date','species_id', 'name_common', 'name_latin', 'radius', 'score', 'lat','long','iNaturalist']
    records = records[keep_columns].sort_values(by=['eco_date'])
    records['eco_date'] = pd.to_datetime(records['eco_date'])
    # filtering out observations older than 5 years
    records = records[records['eco_date'] >= (pd.Timestamp.now() - pd.DateOffset(years=5))]
    insert_log_entry('Observations < 5 years old:', str(len(records)))
    insert_log_entry('Observations WITHOUT iNaturalist:', str(records['iNaturalist'].isna().sum()))
    insert_log_entry('Observations used:', str(len(records)))
    return records

def observations_to_circles(records, default_crs=4326, buffer_crs=6262):
    # Convert the DataFrame to a GeoDataFrame
    records = gpd.GeoDataFrame(records, geometry=gpd.points_from_xy(records.long, records.lat), crs=f"EPSG:{default_crs}")

    # Convert to a projected CRS (in this case using one suitable for Colombia)
    records = records.to_crs(epsg=buffer_crs)

    # Buffer to create circles using the radius column
    records['geometry'] = records.apply(lambda row: row['geometry'].buffer(row['radius'] * 1000), axis=1)

    # If you want to convert back to EPSG:4326
    records = records.to_crs(epsg=default_crs)
    return records

def expand_observations(observations):
    # Expand each row 60 times
    obs_expanded = observations.loc[observations.index.repeat(60)].reset_index(drop=True)

    # Assign 'day' values ranging from -29 to 30
    obs_expanded['day'] = list(range(-29, 31)) * len(observations['eco_id'])

    # Calculate the 'new_date' column
    obs_expanded['eco_date'] = pd.to_datetime(obs_expanded['eco_date'])
    obs_expanded['date'] = obs_expanded['eco_date'] + pd.to_timedelta(obs_expanded['day'], unit='D')
    return obs_expanded

def venn_decomposition(polygons, scores):
    """Decompose a list of polygons like a Venn diagram."""
    
    all_geoms_and_scores = []
    
    while polygons:
        # Step 1: Group polygons by their geometry string representation
        gdf = gpd.GeoDataFrame({'geometry': polygons, 'score': scores})
        gdf['geometry_wkt'] = gdf['geometry'].apply(lambda geom: geom.wkt)
        #gdf['geometry_wkt'] = gdf['geometry'].apply(sorted_wkt)

        # Step 2: For every group choose only the one with max score value and discard the rest
        grouped = gdf.groupby('geometry_wkt').agg({'score': 'max'}).reset_index()
        grouped['geometry'] = grouped['geometry_wkt'].apply(wkt.loads)
        grouped.drop('geometry_wkt', axis=1, inplace=True)
        grouped = gpd.GeoDataFrame(grouped, geometry='geometry')
        grouped = grouped.dissolve(by='score').reset_index()
        grouped['area'] = grouped['geometry'].apply(lambda geom: geom.area)
        grouped = grouped.sort_values(by='area', ascending=False)
        #print('grouping:', len(gdf), len(grouped))
        
        polygons = grouped['geometry'].tolist()
        scores = grouped['score'].tolist()
        
        #print('before pop:', len(polygons), scores, [p.area for p in polygons])
        # Start with the first polygon and score
        first_poly, first_score = polygons.pop(0), scores.pop(0)

        new_polygons = []
        new_scores = []

        for poly, score in zip(polygons, scores):
            # Compute the intersection of poly with first_poly
            intersect = first_poly.intersection(poly)
            if not intersect.is_empty:
                # Append the intersection part with the score of the first_poly
                new_polygons.append(intersect)
                new_scores.append(first_score)
                # Append the intersection part with the score of the current polygon
                new_polygons.append(intersect)
                new_scores.append(score)

            # Compute the difference part of poly
            diff = poly.difference(first_poly)
            if not diff.is_empty:
                new_polygons.append(diff)
                new_scores.append(score)

        # Compute the difference part of first_poly against the union of all other polygons
        polygons = new_polygons
        scores = new_scores

        if not polygons:
            first_diff = first_poly
        else:
            first_diff = first_poly.difference(gpd.GeoSeries(polygons).unary_union)
        if not first_diff.is_empty:
            all_geoms_and_scores.append((first_diff, first_score))

        #print('after extend:', len(polygons), scores, [p.area for p in polygons])
    
    # Convert results to a GeoDataFrame
    gdf_result = gpd.GeoDataFrame(all_geoms_and_scores, columns=['geometry', 'score'])
    
    # Dissolve polygons by score to produce MultiPolygons
    dissolved_gdf = gdf_result.dissolve(by='score').reset_index()

    return dissolved_gdf

def nonoverlapping_maxscore(obs):
    polys = obs['geometry'].tolist()
    scores = obs['score'].tolist()
    result = venn_decomposition(polys, scores)
    return result

def daily_score_union(eco_expanded):
    eco_score = eco_expanded.groupby('date').apply(lambda group: nonoverlapping_maxscore(group)).reset_index()
    eco_score = gpd.GeoDataFrame(eco_score, geometry='geometry', crs=eco_expanded.crs)
    return eco_score
    
def put_missing_dates(union_gdf):
    date_range = pd.date_range(start=union_gdf['date'].min(), end=union_gdf['date'].max())
    date_df = pd.DataFrame(date_range, columns=['date'])
    merged_gdf = date_df.merge(union_gdf, on='date', how='left')
    merged_gdf['geometry'] = merged_gdf['geometry'].fillna(Polygon())
    merged_gdf = gpd.GeoDataFrame(merged_gdf, geometry='geometry', crs=union_gdf.crs)
    return merged_gdf

def daily_video(daily_score, lands, first_date=None):
    if first_date is None:
        first_date = daily_score['date'].min()
    eco_score = daily_score.query(f'date>"{first_date}"').copy()
    eco_score = put_missing_dates(eco_score).to_crs(epsg=3857)
    fincas3857 = lands.to_crs(epsg=3857)
    gdf_list = [group for _, group in eco_score.groupby('date')]
    gdf_list.sort(key=lambda x: x['date'].iloc[0])
    fig, ax = plt.subplots(figsize=(10, 10))
    frame_rate = 30  # frames per second

    def add_basemap(ax, zoom=10, source=ctx.providers.Esri.NatGeoWorldMap): 
        xmin, xmax, ymin, ymax = ax.axis()
        try:
            basemap, extent = ctx.bounds2img(xmin, ymin, xmax, ymax, zoom=zoom, source=source)
            ax.imshow(basemap, extent=extent, interpolation='bilinear')
            ax.axis((xmin, xmax, ymin, ymax))
        except ValueError:
            print(f"Error encountered at zoom level {zoom}. Trying a different zoom level...")
            add_basemap(ax, zoom=zoom-1, source=source)  # decrement zoom level and retry

    def update(num):
        if num%100 == 0:
            print(f"Processing frame {num} with index {num}")
        ax.clear()
        npmsp = gdf_list[num]
        date = npmsp['date'].iloc[0]
        npmsp = npmsp[~npmsp['geometry'].is_empty]

        if len(npmsp) == 0:
            pass
        else:
            npmsp.boundary.plot(ax=ax, color='blue')
            npmsp.plot(ax=ax, column='score', legend=False, cmap='tab10', alpha=0.5, vmin=0.5, vmax=1)

        ax.set_title(f"Date: {date.strftime('%Y-%m-%d')}")
        ax.set_xlim(eco_score.total_bounds[0], eco_score.total_bounds[2])
        ax.set_ylim(eco_score.total_bounds[1], eco_score.total_bounds[3])
        fincas3857.boundary.plot(ax=ax, color='green', linewidth=0.8)  # adjust color and other styling as needed
        fincas3857.plot(ax=ax, facecolor='none', edgecolor='green', linewidth=0.8, alpha=0.5) 
        
        # Add the basemap
        add_basemap(ax, zoom=10, source=ctx.providers.Esri.NatGeoWorldMap)

    ani = animation.FuncAnimation(fig, update, frames=len(gdf_list), repeat=False)
    ani.save('raindrops.mp4', writer=animation.FFMpegWriter(fps=frame_rate))

def daily_attibution(eco_score, lands, obs_expanded, crs=6262):
    fincas6262 = lands.to_crs(epsg=crs)
    eco_score6262 = eco_score.to_crs(epsg=crs)
    obs_expanded6262 = obs_expanded.to_crs(epsg=crs)
    result_df = pd.DataFrame()
    fincas_reset = fincas6262.reset_index().rename(columns={'index': 'finca_name'})
    fincas_reset['total_area'] = fincas_reset['geometry'].area / 10000
    # For each unique date and score in eco_score
    for (date, score), group in eco_score6262.groupby(['date', 'score']):
        # Spatial join between group and fincas
        intersections = gpd.overlay(group, fincas_reset, how='intersection')
        
        # Calculate area of intersection in hectares (assuming CRS is in meters)
        intersections['area'] = intersections['geometry'].area / 10000
        
        # Add to result dataframe
        temp_df = pd.DataFrame({
            'date': [date] * len(intersections),
            'finca': intersections['finca_name'],
            'score': [score] * len(intersections),
            'total_area': intersections['total_area'],
            'area_intersect': intersections['area']
        })
        
        result_df = pd.concat([result_df, temp_df])
    result_df['area_score'] = result_df['area_intersect'] * result_df['score']
    # Reset index of the final result
    attribution = result_df.reset_index(drop=True)
    # Convert the 'date' column to a datetime object
    attribution['date'] = pd.to_datetime(attribution['date'])
    attribution.sort_values(by=['date', 'finca','score'], inplace=True)
    # Set the 'date' column as the index
    attribution = attribution.set_index('date')

    # Finding observations for each finca/date/score combination
    fincas_to_obs = gpd.sjoin(fincas6262, obs_expanded6262, how="inner", predicate="intersects")
    # Group by date, farm, and score and aggregate the eco_id into lists
    fincas_to_obs = (fincas_to_obs.groupby([fincas_to_obs.index, 'date', 'score'])
                .agg({'eco_id': list})
                .reset_index())
    fincas_to_obs.rename(columns={'level_0': 'finca'}, inplace=True)
    attribution = attribution.merge(fincas_to_obs, 
                               left_on=['date', 'finca', 'score'], 
                               right_on=['date', 'finca', 'score'], 
                               how='left')
    attribution = attribution.set_index('date')
    return attribution

def monthly_attribution(attribution):
    # Group by month and finca, sum the scores, and divide by the constant
    attr_month = (attribution.groupby([pd.Grouper(freq='M'), 'finca'])
            .agg({'total_area': 'first', 'area_score': 'sum','eco_id': lambda x: sorted(list(set(sum(x, []))))}) #'eco_id': lambda x: list(set(x))
            .reset_index())
    attr_month['credits'] = attr_month['area_score'] * (1/60)
    attr_month.sort_values(by=['date', 'finca'], inplace=True)
    return attr_month

def cummulative_attribution(attr_month, cutdays= 30, start_date = None):
    a = attr_month.copy()
    if start_date is None:
        start_date = a['date'].min() - pd.DateOffset(days=1)
    else:
        start_date = pd.Timestamp(start_date)  # Replace with your specific date
    mask = (a['date'] < (pd.Timestamp.now() - pd.DateOffset(days=cutdays))) & (a['date'] > start_date)
    a = a[mask]
    a = a.groupby('finca').agg({'date':'last','total_area':'first','credits':'sum', 'eco_id':lambda x: sorted(list(set(sum(x, []))))})
    a.reset_index(inplace=True)
    return a

def delete_all_records_from_airtable(HEADERS, API_URL):
    # Initialize an empty list to collect all record ids
    all_record_ids = []

    # First fetch to initialize pagination
    response = requests.get(API_URL, headers=HEADERS)
    if response.status_code != 200:
        print("Error fetching record IDs:", response.text)
        return False

    records = response.json().get('records', [])
    all_record_ids.extend([record['id'] for record in records])
    
    # Continue fetching records until we've got them all
    while 'offset' in response.json():
        offset = response.json().get('offset')
        response = requests.get(f"{API_URL}?offset={offset}", headers=HEADERS)
        records = response.json().get('records', [])
        all_record_ids.extend([record['id'] for record in records])

    # Delete the records using their IDs
    for record_id in all_record_ids:
        del_response = requests.delete(f"{API_URL}/{record_id}", headers=HEADERS)
        time.sleep(0.2)
        if del_response.status_code != 200:
            print(f"Error deleting record {record_id}:", del_response.text)

    return True


def insert_gdf_to_airtable(gdf, table, insert_geo = False, delete_all = False):
    gdf = gdf.copy()
    config = load_config()
    BASE_ID = config['BIOCREDITS-CALC']['BASE_ID']
    PERSONAL_ACCESS_TOKEN = config['PAT_BIOCREDITS-CALC']

    if insert_geo and 'geometry' in gdf.columns:
        gdf['geometry'] = gdf['geometry'].apply(lambda x: x.wkt)
    elif not insert_geo and 'geometry' in gdf.columns:
        gdf.drop(columns=['geometry'], inplace=True)

    for col in gdf.columns:
        if gdf[col].dtype == 'datetime64[ns]':
            gdf[col] = gdf[col].astype(str)
        if gdf[col].dtype == 'O':
            gdf[col] = gdf[col].astype(str)

    gdf.fillna('', inplace=True)
    # Convert GeoDataFrame to list of records
    records = gdf.to_dict('records')

    # API endpoint and headers
    API_URL = f"https://api.airtable.com/v0/{BASE_ID}/{table}"
    HEADERS = {
        "Authorization": f"Bearer {PERSONAL_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    
    if delete_all:
        # Delete all records
        if not delete_all_records_from_airtable(HEADERS, API_URL):
            print("Error deleting records. Aborting insertion.")
            return


    batch_size = 10
    chunks = [records[i:i + batch_size] for i in range(0, len(records), batch_size)]

    for chunk in chunks:
        json_call = {"records": [{"fields": record} for record in chunk]}
        response = requests.post(API_URL, headers=HEADERS, json=json_call)
        time.sleep(0.2)
        if response.status_code != 200:
            print("Error:", response.text)

def trigger_delete_webhook(table):
    config = load_config()
    HEADERS = {"Content-Type": "application/json"}
    url = config["BIOCREDITS-CALC"]["DELETE_TABLE_WEBHOOK"][table]
    response = requests.post(url, headers=HEADERS, data="{}")
    print(response.text)

def insert_log_entry(event, info):
    df = pd.DataFrame({'Event': [event],'Info': [info]})
    insert_gdf_to_airtable(df, "Logs", insert_geo=False, delete_all=False)