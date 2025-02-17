import pandas as pd
import geopandas as gpd
from calc_utils import load_shp, normalize_shps, shp_to_land, download_observations, \
                      observations_to_circles, expand_observations, daily_score_union, \
                      project_monthly_animation

def test_project_animation():
    try:
        print("1. Loading shapefiles...")
        shp = load_shp('SHP/')
        normalized_shapes = normalize_shps(shp)
        gdf_normalized = shp_to_land(normalized_shapes)
        
        print("2. Loading metadata...")
        land_metadata = pd.read_csv('land_metadata.csv')
        
        print("3. Merging data...")
        gdf_normalized = gdf_normalized.reset_index()
        gdf_normalized['plot_id'] = gdf_normalized['index'].astype(str).str.zfill(3)
        land_metadata['plot_id'] = land_metadata['plot_id'].astype(str).str.zfill(3)
        lands = gdf_normalized.merge(land_metadata, on='plot_id', how='left')
        
        print("\nAvailable project_biodiversity IDs:")
        project_ids = lands['project_biodiversity'].dropna().unique()
        print(project_ids)
        
        print("\nDebug Info:")
        print(f"Total lands: {len(lands)}")
        print(f"Lands with project_biodiversity = 2:")
        project_lands = lands[lands['project_biodiversity'] == 2][['plot_id', 'project_biodiversity']]
        print(project_lands)
        
        print("\n4. Downloading observations...")
        records = download_observations()
        print(f"Downloaded {len(records)} observations")
        
        print("5. Processing observations...")
        records = observations_to_circles(records, default_crs=4326, buffer_crs=6262)
        print("Circles created")
        
        print("6. Expanding observations...")
        obs_expanded = expand_observations(records)
        print(f"Expanded to {len(obs_expanded)} records")
        
        print("7. Creating daily score union...")
        daily_score = daily_score_union(obs_expanded)
        print(f"Created {len(daily_score)} daily scores")
        
        # Create animation for project_biodiversity = 2
        project_id = 2  # We know this exists from the output
        print(f"\n8. Creating animation for project_biodiversity = {project_id}")
        
        animation_path, screenshot_path = project_monthly_animation(daily_score, lands, project_id, first_date='2023-01-01')
        
        if animation_path and screenshot_path:
            print(f"Animation saved to: {animation_path}")
            print(f"Screenshot saved to: {screenshot_path}")
        else:
            print(f"Failed to create animation - no lands found for project_biodiversity = {project_id}")
            
    except KeyboardInterrupt:
        print("\nOperation was interrupted")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    test_project_animation() 