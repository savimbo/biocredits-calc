import pytz
import traceback
from datetime import datetime
from calc_utils import clear_biocredits_tables, download_kml_official, kml_to_shp, load_shp, normalize_shps, \
                       reorder_polygons, shp_to_land, plot_land, download_observations, observations_to_circles, \
                       expand_observations, daily_score_union, daily_video, daily_attibution, monthly_attribution, \
                       cummulative_attribution, insert_gdf_to_airtable, insert_log_entry, upload_to_gcs, get_area_certifier, \
                       create_value_lands, plot_value_lands, transform_one_row_per_value, slider_plot, project_buffer_areas, project_buffer_credits, plot_project_credits   
try:
    colombia_tz = pytz.timezone('America/Bogota')
    start_str = datetime.now(colombia_tz).strftime('%Y-%m-%d %H:%M:%S')
    clear_biocredits_tables(["Logs", "Observations", "Monthly Attribution", "Cummulative Attribution"])
    
    # Download KMLs and metadata
    insert_log_entry('Start time', start_str)
    land_metadata = download_kml_official()
    
    # KML to SHP
    kml_to_shp(source_directory='KML/', destination_directory='SHP/', original_shp_directory='SHPoriginal/')
    kml_to_shp(source_directory='credit_subtypes/KML/', destination_directory='credit_subtypes/SHP/', original_shp_directory=None)

    shp = load_shp('SHP/')
    insert_log_entry('Number of fincas', str(len(shp)))

    normalized_shapes = normalize_shps(shp)
    gdf_normalized = shp_to_land(normalized_shapes)
    
    # Merge metadata with the geographic data
    gdf_normalized = gdf_normalized.reset_index()
    gdf_normalized['plot_id'] = gdf_normalized['index'].astype(str).str.zfill(3)
    lands = gdf_normalized.merge(land_metadata, on='plot_id', how='left')
    

    subtypes = load_shp('credit_subtypes/SHP/')
    platinum = subtypes['Tropical Andes']['geometry'][0]
    value_lands, platinum_gdf = create_value_lands(lands, platinum)
    fig = plot_value_lands(value_lands, platinum_gdf, filename='plots_value.html')
    insert_log_entry('Plots with value (platinum, gold):', upload_to_gcs('biocredits-calc', 'plots_value.html', 'plots_value.html'))

    value_counts = value_lands.groupby('plot_id').agg({'value':'unique'})
    value_counts['value_str'] = value_counts['value'].apply(lambda v: ' & '.join(v))
    insert_log_entry('Plots with value types:', str(value_counts['value_str'].value_counts().to_dict()))

    records = download_observations()
    records = observations_to_circles(records, default_crs=4326, buffer_crs=6262)

    insert_gdf_to_airtable(records, 'Observations', insert_geo = False, delete_all=True)

    obs_expanded = expand_observations(records)
    daily_score = daily_score_union(obs_expanded)

    fig = slider_plot(fig, daily_score, obs_expanded, 1,  '2022-01-01','plots_slider.html')
    insert_log_entry('Time slider plot (since 2022):', upload_to_gcs('biocredits-calc', 'plots_slider.html', 'plots_slider.html'))

    # choose one of the following attribution methods 
    # using value_lands or lands
    attribution = daily_attibution(daily_score, value_lands, obs_expanded, crs=6262)
    #attribution = daily_attibution(daily_score, lands, obs_expanded, crs=6262)
    insert_log_entry('Daily Attribution rows:', str(len(attribution)))
    attribution.to_csv('daily_attribution.csv')
    insert_log_entry('Daily attribution csv:', upload_to_gcs('biocredits-calc', 'daily_attribution.csv', 'daily_attribution.csv'))

    attr_month = monthly_attribution(attribution)

    attr_cumm = cummulative_attribution(attr_month, cutdays = 30, start_date=None)

    attr_month = transform_one_row_per_value(attr_month, 'month')
    insert_log_entry('Monthly Attribution rows:', str(len(attr_month)))
    attr_month.to_csv('monthly_attribution.csv')
    insert_log_entry('Monthly attribution csv:', upload_to_gcs('biocredits-calc', 'monthly_attribution.csv', 'monthly_attribution.csv'))


    attr_cumm = transform_one_row_per_value(attr_cumm, 'cumm')
    insert_log_entry('Cummulative Attribution rows:', str(len(attr_cumm)))
    attr_cumm.to_csv('cummulative_attribution.csv')
    insert_log_entry('Cummulative attribution csv:', upload_to_gcs('biocredits-calc', 'cummulative_attribution.csv', 'cummulative_attribution.csv'))

    pbc_buffer, pbc_union = project_buffer_areas(lands)
    project_credits = project_buffer_credits(pbc_buffer, pbc_union, daily_score, obs_expanded)
    project_credits.to_csv('project_credits.csv')
    insert_log_entry('Project credits csv:', upload_to_gcs('biocredits-calc', 'project_credits.csv', 'project_credits.csv'))
    for project_biodiversity in sorted(project_credits['project_biodiversity'].unique().tolist()):
        credits_fig, ratio_fig = plot_project_credits(project_credits, project_biodiversity)
        insert_log_entry(f'Leakage Report credits plot project: {project_biodiversity}:', upload_to_gcs('biocredits-calc', f'leakage_report_credits_{project_biodiversity}.html', f'leakage_report_credits_{project_biodiversity}.html'))
        insert_log_entry(f'Leakage Report ratio project: {project_biodiversity}:', upload_to_gcs('biocredits-calc', f'leakage_report_ratio_{project_biodiversity}.html', f'leakage_report_ratio_{project_biodiversity}.html'))

    for project_biodiversity in sorted(project_credits['project_biodiversity'].unique().tolist()):
        video_title=f"raindrops_project_{project_biodiversity}.mp4"
        total_bounds = pbc_buffer.query(f'plot_id == "{project_biodiversity}"').to_crs(epsg=3857).total_bounds
        xlim = (total_bounds[0], total_bounds[2])
        ylim = (total_bounds[1], total_bounds[3])
        print(f'total bouds {project_biodiversity}:', total_bounds)
        daily_video(daily_score, lands.query(f'project_biodiversity == "{project_biodiversity}"'), first_date=None, xlim=xlim, ylim=ylim, video_title=video_title)
        insert_log_entry(f'Raindrops Video for project {project_biodiversity}:', upload_to_gcs('biocredits-calc', video_title, video_title))
    
    insert_gdf_to_airtable(attr_cumm.drop(columns=['proportion_certified']), 'Cummulative Attribution', insert_geo = False, delete_all=True)
    insert_gdf_to_airtable(attr_month.drop(columns=['proportion_certified']), 'Monthly Attribution', insert_geo = False, delete_all=True)

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
