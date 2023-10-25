
import time
import pytz
from datetime import datetime
from calc_utils import download_kml_official, kml_to_shp, load_shp, normalize_shps, reorder_polygons, shp_to_land, plot_land, download_observations, observations_to_circles, expand_observations, daily_score_union, daily_video, daily_attibution, monthly_attribution, cummulative_attribution, insert_gdf_to_airtable, trigger_delete_webhook, insert_log_entry

# start time
colombia_tz = pytz.timezone('America/Bogota')
start_str = datetime.now(colombia_tz).strftime('%Y-%m-%d %H:%M:%S')

# deleting old data from airtable
trigger_delete_webhook("Logs")
trigger_delete_webhook("Observations")
trigger_delete_webhook("Daily Attribution")
trigger_delete_webhook("Monthly Attribution")
trigger_delete_webhook("Cummulative Attribution")
time.sleep(10)

# Download KMLs
download_kml_official()
# KML to SHP
kml_to_shp(source_directory='KML/', destination_directory='SHP/')
# Loading SHP files
shp = load_shp('SHP/')

# inserting log entries
insert_log_entry('Start time', start_str)
insert_log_entry('Number of fincas', str(len(shp)))

# processing farm plots
normalized_shapes = normalize_shps(shp)
gdf_normalized = shp_to_land(normalized_shapes)
plot_land(gdf_normalized, 'fincas0.html')
insert_log_entry('Fincas plot:', "Here will be an attachment with the fincas shown without reordering")
# to do: save fincas0.html to airtable
reorder_lands = ['Finca -8  - Luis Lopez (1)', 'Finca -62 -  Enrrique Toro - Official']
insert_log_entry('Reorder:', str(reorder_lands))
normalized_shapes_reordered = reorder_polygons(normalized_shapes, reorder_lands=reorder_lands)
lands = shp_to_land(normalized_shapes_reordered)
plot_land(lands, 'fincas1.html')
insert_log_entry('Fincas plot reordered:', "Here will be an attachment with the fincas shown with reordering")
# to do: save fincas1.html to airtable

# Downloading observations
records = download_observations()
records = observations_to_circles(records, default_crs=4326, buffer_crs=6262)

insert_gdf_to_airtable(records, 'Observations', insert_geo = False, delete_all=True)

# Daily score shapes (union of circles by score)
# here is the logic to transform the observations into a daily shape for each score
obs_expanded = expand_observations(records)
daily_score = daily_score_union(obs_expanded)

# daily video
#daily_video(daily_score, lands, first_date=None)
insert_log_entry('Raindrops Video:', "Here will be an attachment with the video")


# daily attribution to each farm
attribution = daily_attibution(daily_score, lands, obs_expanded, crs=6262)
insert_log_entry('Daily Attribution rows:', str(len(attribution)))
#insert_gdf_to_airtable(attribution.reset_index(), 'Daily Attribution', insert_geo = False, delete_all=True)


# monthly attribution to each farm
attr_month = monthly_attribution(attribution)
insert_log_entry('Monthly Attribution rows:', str(len(attr_month)))
insert_gdf_to_airtable(attr_month, 'Monthly Attribution', insert_geo = False, delete_all=True)

# cummulative attribution to each farm
attr_cumm = cummulative_attribution(attr_month, cutdays = 30, start_date=None)
insert_log_entry('Cummulative Attribution rows:', str(len(attr_cumm)))
insert_gdf_to_airtable(attr_cumm, 'Cummulative Attribution', insert_geo = False, delete_all=True)

# endtime log
end_str = datetime.now(colombia_tz).strftime('%Y-%m-%d %H:%M:%S')
insert_log_entry('End time', end_str)
