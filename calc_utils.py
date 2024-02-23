import os
import requests
import shutil
import json
import time
import subprocess
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon, GeometryCollection
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
    config = load_config()
    BASE_ID = config['KML_TABLE']['BASE_ID']
    TABLE_NAME = config['KML_TABLE']['TABLE_NAME']
    VIEW_ID = config['KML_TABLE']['VIEW_ID']
    FIELD = config['KML_TABLE']['FIELD']
    PERSONAL_ACCESS_TOKEN = config['PERSONAL_ACCESS_TOKEN']

    SAVE_DIRECTORY = save_directory
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

    good_plots = 0
    for record in all_records:
        field = record['fields'].get(FIELD)
        if field:
            url = field[0]['url']
            plot_id = str(record['fields'].get('plot_id'))
            plot_id = f"{plot_id:0>3}"
            save_path = os.path.join(SAVE_DIRECTORY, plot_id+'.kml')
            
            with requests.get(url, stream=True) as file_response:
                with open(save_path, 'wb') as file:
                    for chunk in file_response.iter_content(chunk_size=8192):
                        file.write(chunk)
            good_plots += 1
            print(f"Downloaded plot_id {plot_id}")   
    insert_log_entry('Total KMLs downloaded:', str(good_plots))

def kml_to_shp(source_directory='KML/', destination_directory='SHP/'):
    # Ensure the destination directory exists
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
    else:
        shutil.rmtree(destination_directory)
        os.makedirs(destination_directory)
    # Loop through all .kml files in the source directory
    error_list = []
    for filename in os.listdir(source_directory):
        if filename.endswith('.kml'):
            base_name = os.path.splitext(filename)[0]  # Get the file name without extension
            print("########## Converting", base_name, "##########")
            source_file_path = os.path.join(source_directory, filename)
            destination_file_path = os.path.join(destination_directory, base_name + '.shp')
            # Convert KML to SHP using ogr2ogr
            cmd = ['ogr2ogr', '-f', 'ESRI Shapefile', destination_file_path, source_file_path]
            subprocess.run(cmd)
            # check if the file was created
            if not os.path.exists(destination_file_path):
                error_list.append(base_name)
                print(f"Error converting {filename} to {base_name}.shp")
            else:
                print(f"Converted {filename} to {base_name}.shp")
    insert_log_entry('Error in plots:', ', '.join(error_list))

def load_shp(directory='SHP/'):
    # Loop through all .shp files in the directory and load them
    gdfs = {}
    for filename in sorted(os.listdir(directory)):
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

def normalize_shps(gdfs):
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
    lands['plot_id'] = lands.index
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
        0.5: "rgba(75, 175, 75, 0.5)",   # Further lightening
        0.4: "rgba(150, 250, 150, 0.5)", # Lightest green
    }
    orange_colorscale = {
        1.0: "rgba(255, 140, 0, 0.7)",  # Dark orange
        0.9: "rgba(255, 165, 0, 0.7)",  # Slightly less dark
        0.5: "rgba(255, 200, 0, 0.7)",  # Mid-transition
        0.4: "rgba(255, 220, 0, 0.7)",  # Lightest orange
    }

    for date, group in date_groups:
        geojson = group.__geo_interface__
        for i, row in group.iterrows():
            hover_text = f"Score: {row['score']:.2f}"
            color_scale = [[0, orange_colorscale[row['score']]], [1, orange_colorscale[row['score']]]] 
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

def fetch_linked_record_name(record_id, headers, cache, AIRTABLE_ENDPOINT):
    if record_id in cache:
        return cache[record_id]
    response = requests.get(f"{AIRTABLE_ENDPOINT}/{record_id}", headers=headers)
    if response.status_code == 200:
        data = response.json()
        common_name = data['fields'].get('species_name_common_es')
        cache[record_id] = common_name
        return common_name
    else:
        print(f"Error fetching record {record_id}:", response.text)
        return None

def download_observations():
    """
    Download KML files from Airtable
    """
    config = load_config()
    BASE_ID = config['OBS_TABLE']['BASE_ID']
    TABLE_NAME = config['OBS_TABLE']['TABLE_NAME']
    VIEW_ID = config['OBS_TABLE']['VIEW_ID']
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

    insert_log_entry('Total observations fetched:', str(len(all_records)))

    records = pd.DataFrame([r['fields'] for r in all_records])
    # keep records with NIVEL de CREDITO (from SPECIES (ES))
    records = records[[not r is np.nan for r in records["integrity_score"]]]
    insert_log_entry('Observations with integrity score:', str(len(records)))
    # transform and create columns
    records['species_id'] = records['species_id'].apply(lambda x: x[0] if len(x)==1 else str(x))
    records['name_latin'] = records['name_latin'].apply(lambda x: x[0] if type(x)==list and len(x)==1 else str(x))
    records['score'] = records['integrity_score'].apply(lambda x: max(x) if type(x)==list else x)
    records['radius'] = records['calc_radius'].apply(lambda x: round(max(x),2) if type(x)==list else x)   # using max radius of row
    cache = {}
    records['name_common'] = records['species_name_es'].apply(
        lambda x: fetch_linked_record_name(x[0], headers, cache, AIRTABLE_ENDPOINT) if type(x)==list and len(x)==1 else None)
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
            npmsp.plot(ax=ax, column='score', legend=False, cmap='tab10', alpha=0.5, vmin=daily_score['score'].min(), vmax=daily_score['score'].max())

        ax.set_title(f"Date: {date.strftime('%Y-%m-%d')}")
        ax.set_xlim(eco_score.total_bounds[0], eco_score.total_bounds[2])
        ax.set_ylim(eco_score.total_bounds[1], eco_score.total_bounds[3])
        fincas3857.boundary.plot(ax=ax, color='green', linewidth=0.8)  # adjust color and other styling as needed
        fincas3857.plot(ax=ax, facecolor='none', edgecolor='green', linewidth=0.8, alpha=0.5) 
        
        # Add the basemap
        add_basemap(ax, zoom=10, source=ctx.providers.Esri.NatGeoWorldMap)
        ax.set_xticks([])
        ax.set_yticks([])

    ani = animation.FuncAnimation(fig, update, frames=len(gdf_list), repeat=False)
    ani.save('raindrops.mp4', writer=animation.FFMpegWriter(fps=frame_rate))

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
    attr_month['credits_all'] = attr_month['area_score'] * (1/60)
    attr_month.sort_values(by=groupcols, inplace=True, ascending=[False, True, False] if with_value else [False, True])
    attr_month.plot_id = attr_month.plot_id.astype(int)
    attr_month['eco_id_list'] = attr_month['eco_id']
    attr_month['eco_id'] = attr_month['eco_id'].apply(lambda x: ', '.join([str(i) for i in x]))
    attr_month['calc_index'] = attr_month.apply(lambda x: str(x.plot_id) + '-' + month_dict[x.date.month] + '-' + str(x.date.year), axis=1)
    attr_month.columns = ['calc_date','plot_id'] + (['value'] if with_value else []) + ['total_area', 'area_score', 'eco_id', 'credits_all', 'eco_id_list', 'calc_index']
    attr_month = attr_month[['calc_index', 'calc_date', 'plot_id'] + (['value'] if with_value else []) + ['total_area', 'credits_all', 'eco_id_list', 'eco_id']]
    return attr_month

def cummulative_attribution(attr_month, cutdays= 30, start_date = None):
    with_value = 'value' in attr_month.columns
    groupcols = ['plot_id', 'value'] if with_value else ['plot_id']
    a = attr_month.copy().sort_values(by=['calc_date']+groupcols, ascending=[True, True, False] if with_value else [True, True])
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
    a.sort_values(by=groupcols, inplace=True, ascending=[True, False] if with_value else [True])
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
        keys = ['calc_date','plot_id','total_area','area_certifier', 'proportion_certified']
    elif mode == 'cumm':
        grouper = 'plot_id'
        final_sort = ('plot_id',True)
        keys = ['first_date', 'last_date', 'total_area','area_certifier', 'proportion_certified']
    
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

