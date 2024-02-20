import pytz
import traceback
from datetime import datetime
from calc_utils import clear_biocredits_tables, download_kml_official, kml_to_shp, load_shp, normalize_shps, \
                       reorder_polygons, shp_to_land, plot_land, download_observations, observations_to_circles, \
                       expand_observations, daily_score_union, daily_video, daily_attibution, monthly_attribution, \
                       cummulative_attribution, insert_gdf_to_airtable, insert_log_entry, upload_to_gcs, get_area_certifier
try:
    colombia_tz = pytz.timezone('America/Bogota')
    start_str = datetime.now(colombia_tz).strftime('%Y-%m-%d %H:%M:%S')
    clear_biocredits_tables(["Logs", "Observations", "Monthly Attribution", "Cummulative Attribution"])
    insert_log_entry('Start time', start_str)

    # Download KMLs
    download_kml_official()
    # KML to SHP
    kml_to_shp(source_directory='KML/', destination_directory='SHP/')

    shp = load_shp('SHP/')
    insert_log_entry('Number of fincas', str(len(shp)))

    normalized_shapes = normalize_shps(shp)
    gdf_normalized = shp_to_land(normalized_shapes)
    plot_land(gdf_normalized, 'fincas0.html')
    insert_log_entry('Fincas plot:', upload_to_gcs('biocredits-calc', 'fincas0.html', 'plots_without_reordering.html'))
    reorder_lands = [] 
    insert_log_entry('Reorder:', str(reorder_lands))
    normalized_shapes_reordered = reorder_polygons(normalized_shapes, reorder_lands=reorder_lands)
    lands = shp_to_land(normalized_shapes_reordered)
    plot_land(lands, 'fincas1.html')
    insert_log_entry('Fincas plot reordered:', upload_to_gcs('biocredits-calc', 'fincas1.html', 'plots_with_reordering.html'))

    records = download_observations()
    records = observations_to_circles(records, default_crs=4326, buffer_crs=6262)

    insert_gdf_to_airtable(records, 'Observations', insert_geo = False, delete_all=True)

    obs_expanded = expand_observations(records)
    daily_score = daily_score_union(obs_expanded)

    attribution = daily_attibution(daily_score, lands, obs_expanded, crs=6262)
    insert_log_entry('Daily Attribution rows:', str(len(attribution)))

    area_cert = get_area_certifier()

    attr_month = monthly_attribution(attribution)
    attr_month = attr_month.merge(area_cert, on='plot_id', how='left')
    attr_month['proportion_certified'] = attr_month.apply(lambda row: min(1,row['area_certifier']/row['total_area']), axis=1)
    attr_month['credits_certified'] = attr_month['credits_all'] * attr_month['proportion_certified']
    attr_month['credits_imrv'] = attr_month['credits_all'] * (1 - attr_month['proportion_certified'])
    insert_log_entry('Monthly Attribution rows:', str(len(attr_month)))

    attr_cumm = cummulative_attribution(attr_month, cutdays = 30, start_date=None)
    attr_cumm = attr_cumm.merge(area_cert, on='plot_id', how='left')
    attr_cumm['proportion_certified'] = attr_cumm.apply(lambda row: min(1,row['area_certifier']/row['total_area']), axis=1)
    attr_cumm['credits_certified'] = attr_cumm['credits_all'] * attr_cumm['proportion_certified']
    attr_cumm['credits_imrv'] = attr_cumm['credits_all'] * (1 - attr_cumm['proportion_certified'])
    insert_log_entry('Cummulative Attribution rows:', str(len(attr_cumm)))

    insert_gdf_to_airtable(attr_cumm.drop(columns=['eco_id_list','proportion_certified']), 'Cummulative Attribution', insert_geo = False, delete_all=True)
    insert_gdf_to_airtable(attr_month.drop(columns=['eco_id_list','proportion_certified']), 'Monthly Attribution', insert_geo = False, delete_all=True)

    daily_video(daily_score, lands, first_date=None)
    insert_log_entry('Raindrops Video:', upload_to_gcs('biocredits-calc', 'raindrops.mp4', 'raindrops.mp4'))

    end_str = datetime.now(colombia_tz).strftime('%Y-%m-%d %H:%M:%S')
    insert_log_entry('End time', end_str)
except Exception as e:
    error_traceback = traceback.format_exc()
    insert_log_entry('Error', str(e)) 
    insert_log_entry("Type of exception:", type(e).__name__)
    insert_log_entry("Traceback", error_traceback)
    print('Error', str(e))
    print("Type of exception:", type(e).__name__)
    print(error_traceback)