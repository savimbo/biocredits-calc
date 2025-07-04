import os
import requests
import shutil
import json
import time
import subprocess
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon, GeometryCollection, LineString
import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.io import to_json
from shapely import wkt
import contextily as ctx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from google.cloud import storage
from collections import defaultdict
import zipfile
import traceback
from plotly.subplots import make_subplots

def load_config():
    with open('config.json', 'r') as f:
        return json.load(f)

def download_kml_official(save_directory='KML/', save_shp_directory='SHPoriginal/'):
    """
    Download KML files and shapefiles from Airtable, and additional metadata.
    Only process rows that have either KML or shapefile data.
    """
    # Create directories if they don't exist
    for directory in [save_directory, save_shp_directory]:
        if not os.path.exists(directory):
            os.makedirs(directory)
        else:
            shutil.rmtree(directory)
            os.makedirs(directory)

    config = load_config()
    BASE_ID = config['KML_TABLE']['BASE_ID']
    TABLE_NAME = config['KML_TABLE']['TABLE_NAME']
    VIEW_ID = config['KML_TABLE']['VIEW_ID']
    FIELD = config['KML_TABLE']['FIELD']
    PERSONAL_ACCESS_TOKEN = config['PERSONAL_ACCESS_TOKEN']

    AIRTABLE_ENDPOINT = f"https://api.airtable.com/v0/{BASE_ID}/{TABLE_NAME}"
    headers = {
        "Authorization": f"Bearer {PERSONAL_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }

    # Create caches for POD and project_biodiversity lookups
    pod_cache = {}
    proj_bio_cache = {}

    all_records = []
    offset = None
    while True:
        params = {'view':VIEW_ID}
        if offset:
            params["offset"] = offset

        response = requests.get(AIRTABLE_ENDPOINT, headers=headers, params=params)
        response_json = response.json()
        
        records = response_json.get('records')
        all_records.extend(records)
        
        offset = response_json.get('offset')
        if not offset:
            break

    # Create a list to store metadata
    metadata = []
    good_plots = 0
    shp_downloaded = 0
    total_records = 0
    
    for record in all_records:
        fields = record['fields']
        kml_field = fields.get(FIELD)
        shapefile = fields.get('shapefile_polygon')
        
        # Skip if neither KML nor shapefile is available
        if not kml_field and not shapefile:
            continue
            
        total_records += 1
        plot_id = str(fields.get('plot_id'))
        plot_id = f"{plot_id:0>3}"
        
        # Download KML if available
        if kml_field:
            url = kml_field[0]['url']
            save_path = os.path.join(save_directory, plot_id+'.kml')
            
            with requests.get(url, stream=True) as file_response:
                with open(save_path, 'wb') as file:
                    for chunk in file_response.iter_content(chunk_size=8192):
                        file.write(chunk)
            good_plots += 1
            print(f"Downloaded KML for plot_id {plot_id}")

        # Download and extract shapefile if available
        if shapefile:
            url = shapefile[0]['url']
            # Create a directory for this plot's shapefile
            plot_shp_dir = os.path.join(save_shp_directory, plot_id)
            if not os.path.exists(plot_shp_dir):
                os.makedirs(plot_shp_dir)
            
            # Download the zip file
            zip_path = os.path.join(plot_shp_dir, f"{plot_id}.zip")
            with requests.get(url, stream=True) as file_response:
                with open(zip_path, 'wb') as file:
                    for chunk in file_response.iter_content(chunk_size=8192):
                        file.write(chunk)
            
            # Extract the zip file
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(plot_shp_dir)
                print(f"Downloaded and extracted shapefile for plot_id {plot_id}")
                # Remove the zip file after extraction
                os.remove(zip_path)
                shp_downloaded += 1
            except zipfile.BadZipFile:
                print(f"Error: Invalid zip file for plot_id {plot_id}")
                continue

        # Fetch actual values for POD and project_biodiversity
        pod_id = fields.get('POD', [''])[0] if isinstance(fields.get('POD'), list) else fields.get('POD', '')
        proj_bio_id = fields.get('project_biodiversity', [''])[0] if isinstance(fields.get('project_biodiversity'), list) else fields.get('project_biodiversity', '')

        pod_name = fetch_linked_record_name(pod_id, headers, pod_cache, AIRTABLE_ENDPOINT, 'CODE') if pod_id else ''
        proj_bio_name = fetch_linked_record_name(proj_bio_id, headers, proj_bio_cache, AIRTABLE_ENDPOINT, 'project_id') if proj_bio_id else ''

        # Collect metadata with actual values
        metadata.append({
            'plot_id': plot_id.zfill(3),
            'POD': pod_name,
            'project_biodiversity': proj_bio_name,
            'area_certifier': fields.get('area_certifier', 0)
        })

    # Save metadata to DataFrame
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv('land_metadata.csv', index=False)
    
    # Add logging for linked records
    insert_log_entry('Unique PODs found:', str(metadata_df['POD'].value_counts(dropna=False).to_dict()))
    insert_log_entry('Unique Project Biodiversity found:', str(metadata_df['project_biodiversity'].value_counts(dropna=False).to_dict()))
    
    insert_log_entry('Total records with KML or shapefile:', str(total_records))
    insert_log_entry('Total KMLs downloaded:', str(good_plots))
    insert_log_entry('Total shapefiles downloaded:', str(shp_downloaded))
    return metadata_df

def kml_to_shp(source_directory='KML/', destination_directory='SHP/', original_shp_directory='SHPoriginal/', verbose=False):
    # Ensure the destination directory exists
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
    else:
        shutil.rmtree(destination_directory)
        os.makedirs(destination_directory)

    # First, move original shapefiles to the destination directory
    if original_shp_directory is not None:
        for plot_id_folder in os.listdir(original_shp_directory):
            plot_id_path = os.path.join(original_shp_directory, plot_id_folder)
            if os.path.isdir(plot_id_path):
                # Skip __MACOSX folders
                if plot_id_folder == '__MACOSX':
                    continue
            
            # Create a directory for this plot's shapefile in the destination
            plot_shp_dir = os.path.join(destination_directory, plot_id_folder)
            if not os.path.exists(plot_shp_dir):
                os.makedirs(plot_shp_dir)
            
            # Find the actual shapefile folder (skipping __MACOSX)
            for subfolder in os.listdir(plot_id_path):
                subfolder_path = os.path.join(plot_id_path, subfolder)
                if os.path.isdir(subfolder_path) and not subfolder.startswith('__MACOSX'):
                    # Move and rename files
                    for file in os.listdir(subfolder_path):
                        if file.endswith(('.cpg', '.shp', '.shx', '.dbf', '.prj')):
                            src_file_path = os.path.join(subfolder_path, file)
                            extension = os.path.splitext(file)[1]
                            dest_file_path = os.path.join(plot_shp_dir, f'{plot_id_folder}{extension}')
                            shutil.copy2(src_file_path, dest_file_path)
                    print(f"Moved original shapefile for plot_id {plot_id_folder}")

    # Then convert KMLs to SHP only if no original shapefile exists
    error_list = []
    for filename in os.listdir(source_directory):
        if filename.endswith('.kml'):
            base_name = os.path.splitext(filename)[0]  # Get the file name without extension
            plot_shp_dir = os.path.join(destination_directory, base_name)
            
            # Skip if we already have the original shapefile
            if os.path.exists(plot_shp_dir) and any(f.endswith('.shp') for f in os.listdir(plot_shp_dir)):
                print(f"Skipping KML conversion for {base_name} - original shapefile exists")
                continue
                
            # Create directory if it doesn't exist
            if not os.path.exists(plot_shp_dir):
                os.makedirs(plot_shp_dir)
                
            print(f"########## Converting {base_name} ##########")
            source_file_path = os.path.join(source_directory, filename)
            destination_file_path = os.path.join(plot_shp_dir, base_name + '.shp')
            
            # Convert KML to SHP using ogr2ogr
            cmd = ['ogr2ogr', '-f', 'ESRI Shapefile', destination_file_path, source_file_path]
            subprocess.run(cmd)
            
            # Check if the file was created
            if not os.path.exists(destination_file_path):
                error_list.append(base_name)
                print(f"Error converting {filename} to {base_name}.shp")
            else:
                print(f"Converted {filename} to {base_name}.shp")
    if verbose:
        insert_log_entry('Error in plots:', ', '.join(error_list))

def load_shp(directory='SHP/'):
    # Loop through all subdirectories in the SHP directory
    gdfs = {}
    for plot_folder in sorted(os.listdir(directory)):
        plot_path = os.path.join(directory, plot_folder)
        if os.path.isdir(plot_path):
            # Look for the .shp file in the plot's folder
            shp_file = os.path.join(plot_path, f"{plot_folder}.shp")
            if os.path.exists(shp_file):
                gdf = gpd.read_file(shp_file)
                gdfs[plot_folder] = gdf
            else:
                print(f"Warning: No shapefile found for plot {plot_folder}")
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
    if len(coord) == 2:
        x, y = coord
        return (x, y, 0)
    x, y, _ = coord
    return (x, y, 0)

def normalize_shps(gdfs, logs=False):
    """
    Given a dictionary of GeoDataFrames:
        1) Convert each geometry to Polygon or MultiPolygon
        2) Ensure coordinates are in lat/long (EPSG:4326)
        3) Return a dictionary of normalized geometries
    """
    lands = {}
    geometry_types_found = {}
    found_crs_count = defaultdict(int)
    
    for key, gdf in gdfs.items():
        if len(gdf['geometry']) == 0:
            print(f"warning: {key} GeoDataFrame is empty!")
            continue

        # Ensure the GeoDataFrame is in EPSG:4326 (lat/long)
        
        if gdf.crs is None:
            #print(f"warning: {key} has no CRS defined, assuming EPSG:4326")
            gdf.set_crs(epsg=4326, inplace=True)
            found_crs_count['no_crs'] += 1
        elif gdf.crs != "EPSG:4326":
            #print(f"warning: {key} has CRS {gdf.crs}, converting to EPSG:4326")
            found_crs_count[str(gdf.crs)] += 1
            gdf = gdf.to_crs(epsg=4326)
        else:
            found_crs_count['EPSG:4326'] += 1

        # Get the first geometry and its type
        first_geom = gdf['geometry'].iloc[0]
        geom_type = first_geom.__class__.__name__
        if geom_type not in geometry_types_found:
            geometry_types_found[geom_type] = []
        geometry_types_found[geom_type].append(key)
        
        try:
            if isinstance(first_geom, (Polygon, MultiPolygon)):
                if len(gdf['geometry']) == 1:
                    lands[key] = first_geom
                else:
                    lands[key] = MultiPolygon(gdf['geometry'].tolist())

            elif isinstance(first_geom, Point):
                # Use all points to create a polygon
                points = [(geom.x, geom.y) for geom in gdf['geometry']]
                if len(points) >= 3:  # Need at least 3 points to make a polygon
                    lands[key] = Polygon(points)
                else:
                    print(f"warning: {key} has only {len(points)} points, need at least 3 to create a polygon")
                    continue
            elif isinstance(first_geom, LineString):
                # Convert LineString to Polygon if it's closed
                coords = list(first_geom.coords)
                if coords[0] == coords[-1] and len(coords) >= 4:  # Need at least 4 points for a closed polygon (first=last)
                    lands[key] = Polygon(coords)
                else:
                    print(f"warning: {key} LineString is not closed or has too few points")
                    continue
            elif isinstance(first_geom, GeometryCollection):
                # Extract all polygons from the collection
                polygons = [g for g in first_geom.geoms if isinstance(g, (Polygon, MultiPolygon))]
                if polygons:
                    if len(polygons) == 1:
                        lands[key] = polygons[0]
                    else:
                        lands[key] = MultiPolygon(polygons)
                else:
                    print(f"warning: {key} GeometryCollection contains no polygons")
                    continue
            else:
                print(f"warning: {key} has unsupported geometry type: {geom_type}")
                continue
        except Exception as e:
            print(f"error processing {key}: {str(e)}")
            # print traceback
            print(traceback.format_exc())
            continue

    # Log the geometry types found
    for geom_type, plot_ids in geometry_types_found.items():
        if logs:
            insert_log_entry(f'Geometry type {geom_type} found in plots:', ', '.join(plot_ids))
        print(f'Geometry type {geom_type} found in plots:', ', '.join(plot_ids))
    
    if logs:
        insert_log_entry('CRS found in plots:', ', '.join([f'{crs}: {count}' for crs, count in found_crs_count.items()]))
        insert_log_entry('Total plots processed:', str(len(lands)))
    print('CRS found in plots:', ', '.join([f'{crs}: {count}' for crs, count in found_crs_count.items()]))
    print('Total plots processed:', str(len(lands)))
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

def shp_to_land(lands, crs = "EPSG:4326", area_crs = "EPSG:6262"):
    lands = gpd.GeoSeries(lands)
    lands = gpd.GeoDataFrame(lands, columns=['geometry'])
    
    lands.crs = crs
    lands = lands.to_crs(area_crs)
    lands['total_area'] = lands['geometry'].area / 10000
    lands = lands.to_crs(crs)
    return lands

def plot_land(lands, filename='lands.html'):
    centroid = lands[lands['geometry'].is_valid].unary_union.centroid
    fig = px.choropleth_mapbox(lands, geojson=lands.geometry, locations=lands.index, color=lands.index,
                                color_discrete_sequence=["red"], zoom=9.8, center = {"lat": centroid.coords.xy[1][0], "lon": centroid.coords.xy[0][0]},
                                opacity=0.5, labels={'index':'Finca'})
    fig.update_layout(mapbox_style="carto-positron")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.update_layout(showlegend=False)
    fig.update_layout(height=600, width=600)
    fig.write_html(filename)

def create_value_lands(lands, platinum):
    lands['plot_id'] = lands['index']
    platinum_gdf = gpd.GeoDataFrame({'geometry': [platinum]}, crs=lands.crs)
    difference_gdf = gpd.overlay(lands, platinum_gdf, how='difference')
    difference_gdf['value'] = 'gold'
    intersection_gdf = gpd.overlay(lands, platinum_gdf, how='intersection')
    intersection_gdf['value'] = 'platinum'
    value_lands = pd.concat([intersection_gdf, difference_gdf], ignore_index=True)
    return value_lands, platinum_gdf

def plot_value_lands(value_lands, platinum_gdf, filename='lands_value.html'):
    platinum_geojson = platinum_gdf.__geo_interface__

    fig = go.Figure(go.Choroplethmapbox(geojson=platinum_geojson, locations=[0], z=[1],
                                        colorscale=["#54BF59", "#54BF59"],
                                        marker_opacity=0.5, 
                                        marker_line_width=0,
                                        showscale=False,
                                        hoverinfo='none',
                                        hovertemplate='Platinum Value Area (Tropical Andes)<extra></extra>'))

    combined_geojson = value_lands.__geo_interface__
    for i, row in value_lands.iterrows():
        hover_text = f"Plot ID: {row['plot_id']}<br>Value: {row['value']}"
        colorscale = ["#F2B52A", "#F2B52A"] if row['value'] == 'gold' else ["#4A4F4A", "#4A4F4A"]  
        fig.add_trace(go.Choroplethmapbox(geojson=combined_geojson,
                                        locations=[i],  
                                        z=[0],  
                                        colorscale=colorscale,
                                        showscale=False,
                                        hoverinfo='none',  
                                        hovertemplate=hover_text+ '<extra></extra>' 
                                        ))
    fig.update_layout(mapbox_style="carto-positron",
                    mapbox_zoom=9, 
                    mapbox_center={"lat": 0.7, "lon": -77},
                    margin={"r":0,"t":0,"l":0,"b":0},
                    showlegend=False, width=800, height=800)
    fig.write_html(filename)
    return fig

def save_without_animations(fig, filename):
    fig_json = to_json(fig)
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Observations in time</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>

    <div id="PlotDiv"></div>

    <script>
    var plotData = {fig_json}; 
    Plotly.newPlot('PlotDiv', plotData.data, plotData.layout, {{
        displayModeBar: false, 
        showLink: false, 
        transition: {{duration: 0}}, 
        frame: {{redraw: false}}, 
        staticPlot: false
    }});
    </script>

    </body>
    </html>
    """
    with open(filename, "w") as file:
        file.write(html_template)


def interpolate_color(score, color_scale):
    # Ensure the score is between 0.0 and 1.0
    if score < 0.0 or score > 1.0:
        raise ValueError("Score must be between 0.0 and 1.0")

    # Sort the color scale keys
    keys = sorted(color_scale.keys())

    # If the score is exactly on one of the keys, return the corresponding color
    if score in color_scale:
        return color_scale[score]
    
    # print(f" Interpolating for score {score}, len(keys): {len(keys)}, keys: {keys}")
    # error: Interpolating for score 0.3, len(keys): 5, keys: [0.4, 0.5, 0.8, 0.9, 1.0]
    if score < keys[0]:  
        return color_scale[keys[0]]
    if score > keys[-1]:
        return color_scale[keys[-1]]

    if score < keys[0]:
        return color_scale[keys[0]]
    if score > keys[-1]:
        return color_scale[keys[-1]]

    # Find the two keys between which the score falls
    for i in range(len(keys) - 1):
        if keys[i] <= score <= keys[i + 1]:
            lower_key = keys[i]
            upper_key = keys[i + 1]
            break

    print(f" Interpolating for score {score}, len(keys): {len(keys)}, keys: {keys}")

    # Linear interpolation of the RGBA values
    lower_color = color_scale[lower_key]
    upper_color = color_scale[upper_key]

    # Convert the RGBA strings to lists of floats
    lower_rgba = [float(c) for c in lower_color[5:-1].split(',')]
    upper_rgba = [float(c) for c in upper_color[5:-1].split(',')]

    ratio = (score - lower_key) / (upper_key - lower_key)

    interpolated_rgba = [
        lower_rgba[i] + (upper_rgba[i] - lower_rgba[i]) * ratio
        for i in range(3)
    ]

    # Interpolating the alpha value as well
    alpha = lower_rgba[3] + (upper_rgba[3] - lower_rgba[3]) * ratio

    return f"rgba({int(interpolated_rgba[0])}, {int(interpolated_rgba[1])}, {int(interpolated_rgba[2])}, {alpha:.2f})"




def slider_plot(fig, scores_gdf, obs_expanded, n_weeks=1, min_date='2023-01-01',filename='plots_slider.html'):
    base_trace_indices = list(range(len(fig.data)))
    min_date = pd.to_datetime(min_date)
    start_date = max(scores_gdf['date'].min(), min_date)
    end_date = scores_gdf['date'].max()
    date_range = pd.date_range(start=start_date, end=end_date, freq=f'{n_weeks}W')
    
    # Filter the DataFrame for only dates that match the generated date range
    filtered_gdf = scores_gdf[scores_gdf['date'].isin(date_range)]
    eco_filtered_gdf = obs_expanded[obs_expanded['date'].isin(date_range)]
    
    # Group by date after filtering
    date_groups = filtered_gdf.groupby('date')
    
    green_colorscale = {
        1.0: "rgba(0, 100, 0, 0.5)",     # Dark green
        0.9: "rgba(25, 125, 25, 0.5)",   # Slightly lighter
        0.8: "rgba(50, 150, 50, 0.5)",
        0.5: "rgba(75, 175, 75, 0.5)",   # Further lightening
        0.4: "rgba(150, 250, 150, 0.5)", # Lightest green
    }
    orange_colorscale = {
        1.0: "rgba(255, 140, 0, 0.7)",  # Dark orange
        0.9: "rgba(255, 165, 0, 0.7)",  # Slightly less dark
        0.8: "rgba(255, 182, 0, 0.7)",  # Between 0.9 and 0.5
        0.5: "rgba(255, 200, 0, 0.7)",  # Mid-transition
        0.4: "rgba(255, 220, 0, 0.7)",  # Lightest orange
    }

    for date, group in date_groups:
        geojson = group.__geo_interface__
        for i, row in group.iterrows():
            hover_text = f"Score: {row['score']:.2f}"
            color_scale = [[0, interpolate_color(row['score'],orange_colorscale)], [1, interpolate_color(row['score'],orange_colorscale)]] 
            #color_scale = [[0, orange_colorscale[row['score']]], [1, orange_colorscale[row['score']]]]
            score_trace = go.Choroplethmapbox(geojson=geojson, locations=[i], z=[row['score']],
                                              colorscale=color_scale, 
                                              zmin=0, zmax=1,
                                              showscale=False, # Set to True if you want a scale bar
                                              hoverinfo='none',  
                                              hovertemplate=hover_text+ '<extra></extra>',
                                              name=str(date),
                                              marker_line_width=0,
                                              ) 
            score_trace.visible = False
            fig.add_trace(score_trace)

    for date in date_range:
        day_data = eco_filtered_gdf[eco_filtered_gdf['date'] == date]
        if day_data.empty:
            continue

        lats = day_data['lat'].tolist()
        lons = day_data['long'].tolist()
        hover_texts = [
            f"eco id: {row['eco_id']}<br>species: {row['name_common']}<br>score: {row['score']}<br>radius: {row['radius']}<br>eco date: {row['eco_date'].strftime('%Y-%m-%d')}"
            for _, row in day_data.iterrows()
        ]
        
        marker_trace = go.Scattermapbox(
            lat=lats,
            lon=lons,
            mode='markers',
            marker=go.scattermapbox.Marker(size=5, color='#1122D2'),
            text=hover_texts,
            hoverinfo='text',
            name=str(date)
        )
        
        marker_trace.visible = False
        fig.add_trace(marker_trace)
    
    sliders = [{
        'steps': []
    }]
    
    for date in date_range:
        visible_traces = [False] * len(fig.data)
        
        for idx in base_trace_indices:
            visible_traces[idx] = True
        
        # Then, for the current date, set the visibility for specific traces
        for i, trace in enumerate(fig.data):
            if pd.to_datetime(trace.name) == date:
                visible_traces[i] = True  # Make the trace for the current date visible
        
        step = {
            'method': 'update',
            'args': [{'visible': visible_traces},
                    {'title': f"Scores for {date.strftime('%Y-%m-%d')}"}],
            'label': date.strftime('%Y-%m-%d')
        }
        sliders[0]['steps'].append(step)
    fig.update_layout(sliders=sliders)
    save_without_animations(fig, filename)
    return fig

def fetch_linked_record_name(record_id, headers, cache, AIRTABLE_ENDPOINT, field_name='species_name_common_es'):
    """
    Fetch the name of a linked record from Airtable.
    Modified to handle different field names for different tables.
    """
    if record_id in cache:
        return cache[record_id]
    response = requests.get(f"{AIRTABLE_ENDPOINT}/{record_id}", headers=headers)
    if response.status_code == 200:
        data = response.json()
        value = data['fields'].get(field_name)
        cache[record_id] = value
        return value
    else:
        print(f"Error fetching record {record_id}:", response.text)
        return None

def download_observations():
    """
    Download KML files from Airtable
    """
    config = load_config()
    BASE_ID = config['OBS_TABLE']['BASE_ID']
    TABLE_ID = config['OBS_TABLE']['TABLE_ID']
    VIEW_ID = config['OBS_TABLE']['VIEW_ID']
    PERSONAL_ACCESS_TOKEN = config['PERSONAL_ACCESS_TOKEN']
    AIRTABLE_ENDPOINT = f"https://api.airtable.com/v0/{BASE_ID}/{TABLE_ID}"

    headers = {
        "Authorization": f"Bearer {PERSONAL_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }

    all_records = []
    offset = None

    while True:
        params = {'view':VIEW_ID}
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
    records = records[[not r is np.nan for r in records["integrity_score"]]]
    insert_log_entry('Observations with integrity score:', str(len(records)))
    # transform and create columns
    #records['species_id'] = records['species_id'].apply(lambda x: x[0] if type(x)==list else x) # we might need a fetch_linked_record_name
    records['name_latin'] = records['name_latin'].apply(lambda x: x[0] if type(x)==list and len(x)==1 else str(x))
    records['score'] = records['integrity_score'].apply(lambda x: max(x) if type(x)==list else x)
    records['radius'] = records['calc_radius'].apply(lambda x: round(max(x),2) if type(x)==list else x)   # using max radius of row
    cache = {}
    records['name_common'] = records['name_common_es'].apply(lambda x: x[0] if type(x)==list and len(x)==1 else str(x))
    records['name_common_'] = records['species_type'].apply(
        lambda x: fetch_linked_record_name(x[0], headers, cache, AIRTABLE_ENDPOINT) if type(x)==list and len(x)==1 else None)
    # We are assuming one observation per record, so we can use the first element of the list, max radius, etc.

    # filter records with radius > 0 and eco_long < 0
    records = records.query('radius>0')
    insert_log_entry('Observations with radius > 0:', str(len(records)))
    records = records.query('eco_long<0')
    insert_log_entry('Observations with eco_long < 0:', str(len(records)))

    # renaming and keeping columns
    records = records.rename(columns={'# ECO':'eco_id', 'eco_lat':'lat', 'eco_long':'long'})
    keep_columns = ['eco_id','eco_date', 'name_common', 'name_latin', 'radius', 'score', 'lat','long','iNaturalist']
    records = records[keep_columns].sort_values(by=['eco_date'])
    records['eco_date'] = pd.to_datetime(records['eco_date'])
    # filtering out observations older than 5 years
    records = records[records['eco_date'] >= (pd.Timestamp.now() - pd.DateOffset(years=10))]
    insert_log_entry('Observations < 10 years old:', str(len(records)))
    insert_log_entry('Observations WITHOUT iNaturalist:', str(records['iNaturalist'].isna().sum()))
    insert_log_entry('Observations used:', str(len(records)))
    insert_log_entry('Scores seen:', str(list(np.sort(records['score' ].unique())[::-1])))
    insert_log_entry('Radius seen:', str(list(np.sort(records['radius'].unique())[::-1])))
    records.sort_values('eco_date', ascending=False, inplace=True)
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

def convert_to_multipolygon(geometry):
    if isinstance(geometry, GeometryCollection):
        polygons = [geom for geom in geometry.geoms if isinstance(geom, (Polygon, MultiPolygon))]
        return MultiPolygon(polygons)
    elif isinstance(geometry, MultiPolygon):
        return geometry
    elif isinstance(geometry, Polygon):
        return MultiPolygon([geometry])
    else:
        raise ValueError("Geometry is neither a Polygon nor a MultiPolygon nor a GeometryCollection")

def daily_score_union(eco_expanded):
    eco_score = eco_expanded.groupby('date').apply(lambda group: nonoverlapping_maxscore(group)).reset_index()
    eco_score = gpd.GeoDataFrame(eco_score, geometry='geometry', crs=eco_expanded.crs)
    eco_score['geometry'] = eco_score['geometry'].apply(convert_to_multipolygon)
    return eco_score
    
def put_missing_dates(union_gdf):
    date_range = pd.date_range(start=union_gdf['date'].min(), end=union_gdf['date'].max())
    date_df = pd.DataFrame(date_range, columns=['date'])
    merged_gdf = date_df.merge(union_gdf, on='date', how='left')
    merged_gdf['geometry'] = merged_gdf['geometry'].fillna(Polygon())
    merged_gdf = gpd.GeoDataFrame(merged_gdf, geometry='geometry', crs=union_gdf.crs)
    return merged_gdf

def daily_video(
    daily_score,
    lands,
    first_date=None,
    xlim=None,
    ylim=None,
    video_title='raindrops.mp4'
):
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
            npmsp.plot(ax=ax, column='score', legend=False, cmap='tab10', alpha=0.5, vmin=daily_score['score'].min(), vmax=daily_score['score'].max())

        ax.set_title(f"Date: {date.strftime('%Y-%m-%d')}")
        
        # Use provided xlim/ylim if available, otherwise use total_bounds
        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])
        else:
            ax.set_xlim(eco_score.total_bounds[0], eco_score.total_bounds[2])
            
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])
        else:
            ax.set_ylim(eco_score.total_bounds[1], eco_score.total_bounds[3])
            
        fincas3857.boundary.plot(ax=ax, color='green', linewidth=0.8)
        fincas3857.plot(ax=ax, facecolor='none', edgecolor='green', linewidth=0.8, alpha=0.5) 
        
        add_basemap(ax, zoom=10, source=ctx.providers.Esri.NatGeoWorldMap)
        ax.set_xticks([])
        ax.set_yticks([])

    ani = animation.FuncAnimation(fig, update, frames=len(gdf_list), repeat=False)
    ani.save(video_title, writer=animation.FFMpegWriter(fps=frame_rate))

def daily_attibution(eco_score, lands, obs_expanded, crs=6262):
    fincas6262 = lands.to_crs(epsg=crs)
    eco_score6262 = eco_score.to_crs(epsg=crs)
    obs_expanded6262 = obs_expanded.to_crs(epsg=crs)
    result_df = pd.DataFrame()
    fincas_reset = fincas6262.reset_index().rename(columns={'index': 'plot_id'}) if not 'plot_id' in fincas6262.columns else fincas6262

    #fincas_reset['total_area'] = fincas_reset['geometry'].area / 10000
    # For each unique date and score in eco_score
    for (date, score), group in eco_score6262.groupby(['date', 'score']):
        # Spatial join between group and fincas
        intersections = gpd.overlay(group, fincas_reset, how='intersection')
        
        # Calculate area of intersection in hectares (assuming CRS is in meters)
        intersections['area'] = intersections['geometry'].area / 10000
        
        # Add to result dataframe
        temp_df = pd.DataFrame({
            'date': [date] * len(intersections),
            'plot_id': intersections['plot_id'],
            'POD': intersections['POD'] if 'POD' in intersections.columns else None,
            'project_biodiversity': intersections['project_biodiversity'] if 'project_biodiversity' in intersections.columns else None,
            'area_certifier': intersections['area_certifier'] if 'area_certifier' in intersections.columns else None,
            'value': intersections['value'] if 'value' in intersections.columns else None,
            'score': [score] * len(intersections),
            'total_area': intersections['total_area'],
            'area_intersect': intersections['area']
        })
        if not 'value' in intersections.columns:
            temp_df.drop(columns=['value'], inplace=True, errors='ignore')
        
        result_df = pd.concat([result_df, temp_df])
    result_df['area_score'] = result_df['area_intersect'] * result_df['score']
    # Reset index of the final result
    attribution = result_df.reset_index(drop=True)
    # Convert the 'date' column to a datetime object
    attribution['date'] = pd.to_datetime(attribution['date'])
    
    # Set the 'date' column as the index
    attribution = attribution.set_index('date')

    # Finding observations for each finca/date/score combination
    fincas_to_obs = gpd.sjoin(fincas6262, obs_expanded6262, how="inner", predicate="intersects")
    # Group by date, farm, and score and aggregate the eco_id into lists
    groupcols = ['plot_id', 'value', 'date', 'score'] if 'value' in fincas_to_obs.columns else ['plot_id', 'date', 'score']  
    fincas_to_obs = (fincas_to_obs.groupby(groupcols)
                .agg({'eco_id': list})
                .reset_index())
    attribution = attribution.merge(fincas_to_obs, 
                               left_on=groupcols, 
                               right_on=groupcols, 
                               how='left')
    attribution = attribution.set_index('date')
    attribution.sort_values(by=['date', 'plot_id','score'], inplace=True, ascending=[False, True, False])
    return attribution

def monthly_attribution(attribution):
    # Group by month and finca, sum the scores, and divide by the constant 30 to get bimonthly credits
    month_dict = {1: "January",2: "February",3: "March",4: "April",5: "May",6: "June",
        7: "July",8: "August",9: "September",10: "October",11: "November",12: "December"}
    with_value = 'value' in attribution.columns
    groupcols = [pd.Grouper(freq='M'), 'plot_id', 'value'] if with_value else [pd.Grouper(freq='M'), 'plot_id']
    attr_month = (attribution.groupby(groupcols)
            .agg({'total_area': 'first', 'area_score': 'sum',
                    'eco_id': lambda x: sorted(list(set(sum(x, []))))}) 
            .reset_index())
    groupcols = ['date', 'plot_id', 'value'] if with_value else ['date', 'plot_id']
    attr_month['credits_all'] = attr_month['area_score'] * (1/30)
    attr_month.sort_values(by=groupcols, inplace=True, ascending=[False, True, True] if with_value else [False, True])
    attr_month.plot_id = attr_month.plot_id.astype(int)
    attr_month['eco_id_list'] = attr_month['eco_id']
    attr_month['eco_id'] = attr_month['eco_id'].apply(lambda x: ', '.join([str(i) for i in x]))
    attr_month['calc_index'] = attr_month.apply(lambda x: str(x.plot_id) + '-' + month_dict[x.date.month] + '-' + str(x.date.year), axis=1)
    attr_month.columns = ['calc_date','plot_id'] + (['value'] if with_value else []) + ['total_area', 'area_score', 'eco_id', 'credits_all', 'eco_id_list', 'calc_index']

    area_cert = attribution[['plot_id', 'POD', 'project_biodiversity', 'area_certifier']].drop_duplicates()
    area_cert['plot_id'] = area_cert['plot_id'].astype(int)
    attr_month = attr_month.merge(area_cert, on='plot_id', how='left')
    attr_month['area_certifier'] = attr_month['area_certifier'].astype(float)
    attr_month['proportion_certified'] = attr_month.apply(lambda row: min(1,row['area_certifier']/row['total_area']), axis=1)
    attr_month['credits_certified'] = attr_month['credits_all'] * attr_month['proportion_certified']
    attr_month['credits_imrv'] = (attr_month['credits_all'] * (1 - attr_month['proportion_certified'])).apply(lambda x: max(x,0))
    attr_month = attr_month[['calc_index', 'calc_date', 'plot_id', 'POD', 'project_biodiversity', 'area_certifier'] + (['value'] if with_value else []) + ['total_area', 'credits_all', 'eco_id_list', 'eco_id'] + ['proportion_certified', 'credits_certified', 'credits_imrv']]
    return attr_month

def cummulative_attribution(attr_month, cutdays= 30, start_date = None):
    with_value = 'value' in attr_month.columns
    groupcols = ['plot_id', 'POD', 'project_biodiversity', 'area_certifier', 'value'] if with_value else ['plot_id', 'POD', 'project_biodiversity', 'area_certifier']
    a = attr_month.copy().sort_values(by=['calc_date']+groupcols, ascending=[True, True, False, False, False, False] if with_value else [True, True, False, False, False])
    if start_date is None:
        start_date = a['calc_date'].min() 
    else:
        start_date = pd.Timestamp(start_date)  
    mask = (a['calc_date'] < (pd.Timestamp.now() - pd.DateOffset(days=cutdays))) & (a['calc_date'] >= start_date)
    a = a[mask]
    a.sort_values
    a = a.groupby(groupcols).agg({
        'calc_date': ['min', 'max'],
        'total_area': 'first',
        'credits_all': 'sum',
        'eco_id_list': lambda x: sorted(list(set(sum(x, []))))
    })
    a.columns = ['first_date', 'last_date', 'total_area', 'credits_all', 'eco_id_list']
    a['eco_id'] = a['eco_id_list'].apply(lambda x: ', '.join([str(i) for i in x]))
    a.reset_index(inplace=True)
    a.sort_values(by=groupcols, inplace=True, ascending=[True, True, False, False, False] if with_value else [True, False, False, False])
    a['proportion_certified'] = a.apply(lambda row: min(1,row['area_certifier']/row['total_area']), axis=1)
    a['credits_certified'] = a['credits_all'] * a['proportion_certified']
    a['credits_imrv'] = (a['credits_all'] * (1 - a['proportion_certified'])).apply(lambda x: max(x,0))
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
    requests.post(url, headers=HEADERS, data="{}")

def insert_log_entry(event, info):
    df = pd.DataFrame({'Event': [event],'Info': [info]})
    insert_gdf_to_airtable(df, "Logs", insert_geo=False, delete_all=False)


def clear_biocredits_tables(tables):
    for table in tables:
        trigger_delete_webhook(table)
    for table in tables:
        trigger_delete_webhook(table)
    time.sleep(5)

    delete_again = []
    config = load_config()
    BASE_ID = config['BIOCREDITS-CALC']['BASE_ID']
    PERSONAL_ACCESS_TOKEN = config['PAT_BIOCREDITS-CALC']
    for TABLE_NAME in tables:
        AIRTABLE_ENDPOINT = f"https://api.airtable.com/v0/{BASE_ID}/{TABLE_NAME}"
        headers = {
            "Authorization": f"Bearer {PERSONAL_ACCESS_TOKEN}",
            "Content-Type": "application/json"
        }

        response = requests.get(AIRTABLE_ENDPOINT, headers=headers)
        response_json = response.json()
        if len(response_json['records']) > 0:
            delete_again.append(TABLE_NAME)
    if len(delete_again) > 0:
        clear_biocredits_tables(delete_again)

    time.sleep(10)

def create_bucket(storage_client, bucket_name):
    bucket = storage_client.bucket(bucket_name)
    if not bucket.exists():
        storage_client.create_bucket(bucket)
        print(f"Bucket {bucket_name} created.")
    else:
        print(f"Bucket {bucket_name} already exists.")

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client.from_service_account_json('the-savimbo-project-511c079217f8.json')
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    if blob.exists():
        blob.delete()
    blob.cache_control = 'no-cache, no-store, must-revalidate'
    blob.upload_from_filename(source_file_name)
    blob.make_public()
    #print(f"File {source_file_name} uploaded to {destination_blob_name} and is now publicly accessible.")
    return blob.public_url

def get_area_certifier():
    config = load_config()
    BASE_ID = config['KML_TABLE']['BASE_ID']
    TABLE_NAME = config['KML_TABLE']['TABLE_NAME']
    VIEW_ID = config['KML_TABLE']['VIEW_ID']
    FIELD = config['KML_TABLE']['FIELD']
    PERSONAL_ACCESS_TOKEN = config['PERSONAL_ACCESS_TOKEN']
    AIRTABLE_ENDPOINT = f"https://api.airtable.com/v0/{BASE_ID}/{TABLE_NAME}"
    headers = {
        "Authorization": f"Bearer {PERSONAL_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }

    all_records = []
    offset = None
    while True:
        params = {'view':VIEW_ID}
        if offset:
            params["offset"] = offset

        response = requests.get(AIRTABLE_ENDPOINT, headers=headers, params=params)
        response_json = response.json()
        
        records = response_json.get('records')
        all_records.extend(records)
        
        offset = response_json.get('offset')
        if not offset:
            break

    area_cert = []
    for record in all_records:
        a = {}
        a['plot_id'] = record['fields'].get('plot_id')
        a['area_certifier'] = record['fields'].get('area_certifier')
        area_cert.append(a)
    return pd.DataFrame(area_cert).fillna(0)

def transform_one_row_per_value(df, mode):
    result = {}
    if mode == 'month':
        grouper = 'calc_index'
        final_sort = (['calc_date','plot_id'],[False,True])
        keys = ['calc_date','plot_id','total_area','area_certifier', 'proportion_certified', 
               'POD', 'project_biodiversity']  # Added new fields
    elif mode == 'cumm':
        grouper = 'plot_id'
        final_sort = ('plot_id',True)
        keys = ['first_date', 'last_date', 'total_area','area_certifier', 'proportion_certified',
               'POD', 'project_biodiversity']  # Added new fields
    
    grouped = df.groupby(grouper)
    for name, group in grouped:
        group_dict = {key: group[key].iloc[0] for key in keys}

        group_dict['values'] = ' & '.join(group['value'].values)
        for _, row in group.iterrows():
            value = row['value']
            group_dict[f'credits_all_{value}'] = row['credits_all']
            group_dict[f'credits_certified_{value}'] = row['credits_certified']
            group_dict[f'credits_imrv_{value}'] = row['credits_imrv']
            group_dict[f'eco_id_{value}'] = row['eco_id'] 
        
        result[name] = group_dict

    df_one_row_per_value = pd.DataFrame(result).transpose().fillna(0).reset_index().rename(columns={'index': grouper}).sort_values(final_sort[0], ascending=final_sort[1])
    return df_one_row_per_value

def project_buffer_areas(lands, buffer_distance=7000, crs=6262):
    """
    Takes a GeoDataFrame of lands, groups by project_biodiversity, creates a buffer,
    and calculates credits for the buffer area excluding the original lands.
    
    Args:
        lands (GeoDataFrame): Input lands with project_biodiversity column
        buffer_distance (float): Buffer distance in meters (default 7km)
        crs (int): EPSG code for the projection to use for buffer calculation
    
    Returns:
        GeoDataFrame: Buffer areas with credit calculations
    """
    # Convert to the specified CRS for accurate buffer calculation
    lands_proj = lands.to_crs(epsg=crs)
    lands_proj['total_area'] = lands_proj['total_area'].astype(float)
    lands_proj['area_certifier'] = lands_proj['area_certifier'].astype(float)

    
    # Group by project_biodiversity and create union
    project_unions = lands_proj.dissolve(by='project_biodiversity', aggfunc= 'sum')
    
    # Create buffers and calculate differences
    buffer_areas = []
    union_areas = []
    for project_id, row in project_unions.iterrows():
        if pd.isna(project_id) or project_id == '':
            continue
            
        # Create buffer
        buffer = row.geometry.buffer(buffer_distance)
        
        # Subtract original area
        buffer_diff = buffer.difference(row.geometry)
        
        # Create record
        buffer_areas.append({
            'plot_id': project_id,
            'geometry': buffer_diff,
            'total_area': buffer_diff.area / 10000,  # Convert to hectares
            # 'original_area': row.geometry.area / 10000,
            # 'buffer_ratio': (buffer_diff.area / row.geometry.area) if row.geometry.area > 0 else 0,
            'area_certifier': 0
        })

        union_areas.append({
            'plot_id': project_id,
            'geometry': row.geometry,
            'total_area': row.geometry.area / 10000,
            'area_certifier': row.area_certifier
        })
    
    # Create GeoDataFrame from results
    buffer_gdf = gpd.GeoDataFrame(buffer_areas, crs=f"EPSG:{crs}")
    union_gdf = gpd.GeoDataFrame(union_areas, crs=f"EPSG:{crs}")
    
    # Convert back to original CRS (assumed to be 4326)
    buffer_gdf = buffer_gdf.to_crs(epsg=4326)
    union_gdf = union_gdf.to_crs(epsg=4326)
    
    return buffer_gdf, union_gdf

def project_buffer_credits(buffer_gdf, union_gdf, daily_score, obs_expanded):
    buffer_attribution = daily_attibution(daily_score, buffer_gdf, obs_expanded, crs=6262).fillna(0)
    buffer_attr_month = monthly_attribution(buffer_attribution)
    union_attribution = daily_attibution(daily_score, union_gdf, obs_expanded, crs=6262).fillna(0)
    union_attr_month = monthly_attribution(union_attribution)
    union_attr_month = union_attr_month[['calc_index', 'calc_date', 'plot_id', 'total_area', 'credits_all', 'eco_id_list']]
    union_attr_month.columns = ['calc_index', 'calc_date', 'project_biodiversity', 'total_area', 'credits_all', 'eco_id_list']
    buffer_attr_month = buffer_attr_month[['calc_index', 'calc_date', 'plot_id', 'total_area', 'credits_all', 'eco_id_list']]
    buffer_attr_month.columns = ['calc_index', 'calc_date', 'project_biodiversity', 'total_area', 'credits_all', 'eco_id_list']
    union_attr_month['type'] = 'union'
    buffer_attr_month['type'] = 'buffer'
    project_credits = pd.concat([buffer_attr_month, union_attr_month])
    return project_credits

def plot_project_credits(project_credits, project_id):
    """
    Creates two independent figures:
    1. Credits evolution over time with dual y-axes (union left, buffer right)
    2. Bar plot showing buffer/union ratio
    
    Args:
        project_credits (DataFrame): DataFrame with project credits data
        project_id (str): project_biodiversity identifier to plot
    
    Returns:
        tuple: (credits_fig, ratio_fig) The generated figures
    """
    # Filter data for the specific project
    proj_data = project_credits[project_credits['project_biodiversity'] == project_id].copy()
    
    # Create continuous monthly date range
    min_date = proj_data['calc_date'].min()
    max_date = proj_data['calc_date'].max()
    dates = pd.date_range(start=min_date, end=max_date, freq='M')  # 'M' gives end of month dates
    
    # Pivot data for easier plotting
    union_data = proj_data[proj_data['type'] == 'union'].set_index('calc_date')['credits_all'].reindex(dates, fill_value=0)
    buffer_data = proj_data[proj_data['type'] == 'buffer'].set_index('calc_date')['credits_all'].reindex(dates, fill_value=0)
    
    # Calculate ratio
    ratios = buffer_data / union_data.replace(0, float('inf'))
    ratios = ratios.replace(float('inf'), 0)
    
    # Create credits figure
    credits_fig = go.Figure()
    
    # Add union credits (left y-axis)
    credits_fig.add_trace(
        go.Scatter(x=dates, y=union_data,
                  name="Union Credits",
                  line=dict(color='blue', width=2))
    )
    
    # Add buffer credits (right y-axis)
    credits_fig.add_trace(
        go.Scatter(x=dates, y=buffer_data,
                  name="Buffer Credits",
                  line=dict(color='green', width=2),
                  yaxis="y2")
    )
    
    # Update credits figure layout
    credits_fig.update_layout(
        title=f"Leakage Report - Credits Evolution - project_id {project_id}",
        showlegend=True,
        plot_bgcolor='rgba(240,240,240,0.5)',
        hovermode='x unified',
        yaxis=dict(
            title="Union Credits",
            titlefont=dict(color="blue"),
            tickfont=dict(color="blue"),
            showgrid=True,
            gridcolor='white',
            side='left'
        ),
        yaxis2=dict(
            title="Buffer Credits",
            titlefont=dict(color="green"),
            tickfont=dict(color="green"),
            overlaying="y",
            side="right",
            showgrid=False
        ),
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridcolor='white'
        )
    )
    
    # Create ratio figure
    ratio_fig = go.Figure()
    
    # Add ratio bars
    ratio_fig.add_trace(
        go.Bar(x=dates, y=ratios,
               name="Buffer/Union Ratio",
               marker_color='red')
    )
    
    # Update ratio figure layout
    ratio_fig.update_layout(
        title=f"Leakage Report - Buffer/Union Ratio - project_id {project_id}",
        showlegend=True,
        plot_bgcolor='rgba(240,240,240,0.5)',
        yaxis=dict(
            title=dict(
                text="Buffer/Union Ratio",
                font=dict(color="red")
            ),
            tickfont=dict(color="red"),
            showgrid=True,
            gridcolor='white'
        ),
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridcolor='white'
        )
    )
    credits_fig.write_html(f'leakage_report_credits_{project_id}.html')
    ratio_fig.write_html(f'leakage_report_ratio_{project_id}.html')
    return credits_fig, ratio_fig
