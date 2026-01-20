import argparse
from datetime import datetime
import json
import os
import sys
import time

import geopandas as gpd
from geopy.geocoders import Nominatim
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import osmnx as ox
from shapely.geometry import box, LineString, Point
from shapely.ops import polygonize_full, unary_union
from tqdm import tqdm

THEMES_DIR = "themes"
FONTS_DIR = "fonts"
POSTERS_DIR = "posters"

# Geographic constants
METERS_PER_DEGREE_LAT = 111000  # Approximate meters per degree of latitude

# Coastline processing constants
COASTLINE_BUFFER_MULT = 1.5    # Fetch coastlines from larger area for completeness
MIN_POLYGON_AREA = 1e-10       # Minimum area to consider a polygon valid
SEA_MIN_AREA_RATIO = 0.1       # Sea must be at least 10% of bounding box
SEA_MIN_SHARED_BOUNDARY = 0.01 # Minimum shared boundary length with land
ISLAND_MAX_AREA_RATIO = 0.2    # Polygons under 20% of bbox are likely islands


def load_fonts():
    """
    Load Roboto fonts from the fonts directory.
    Returns dict with font paths for different weights.
    """
    fonts = {
        'bold': os.path.join(FONTS_DIR, 'Roboto-Bold.ttf'),
        'regular': os.path.join(FONTS_DIR, 'Roboto-Regular.ttf'),
        'light': os.path.join(FONTS_DIR, 'Roboto-Light.ttf')
    }
    
    # Verify fonts exist
    for weight, path in fonts.items():
        if not os.path.exists(path):
            print(f"⚠ Font not found: {path}")
            return None
    
    return fonts

FONTS = load_fonts()

def generate_output_filename(city, theme_name, output_format='png'):
    """
    Generate unique output filename with city, theme, and datetime.
    """
    if not os.path.exists(POSTERS_DIR):
        os.makedirs(POSTERS_DIR)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    city_slug = city.lower().replace(' ', '_')
    filename = f"{city_slug}_{theme_name}_{timestamp}.{output_format}"
    return os.path.join(POSTERS_DIR, filename)

def get_available_themes():
    """
    Scans the themes directory and returns a list of available theme names.
    """
    if not os.path.exists(THEMES_DIR):
        os.makedirs(THEMES_DIR)
        return []
    
    themes = []
    for file in sorted(os.listdir(THEMES_DIR)):
        if file.endswith('.json'):
            theme_name = file[:-5]  # Remove .json extension
            themes.append(theme_name)
    return themes

def load_theme(theme_name="feature_based"):
    """
    Load theme from JSON file in themes directory.
    """
    theme_file = os.path.join(THEMES_DIR, f"{theme_name}.json")
    
    if not os.path.exists(theme_file):
        print(f"⚠ Theme file '{theme_file}' not found. Using default feature_based theme.")
        # Fallback to embedded default theme
        return {
            "name": "Feature-Based Shading",
            "bg": "#FFFFFF",
            "text": "#000000",
            "gradient_color": "#FFFFFF",
            "water": "#C0C0C0",
            "parks": "#F0F0F0",
            "sea": "#E8F4F8",
            "land": "#FFFFFF",
            "road_motorway": "#0A0A0A",
            "road_primary": "#1A1A1A",
            "road_secondary": "#2A2A2A",
            "road_tertiary": "#3A3A3A",
            "road_residential": "#4A4A4A",
            "road_default": "#3A3A3A"
        }
    
    with open(theme_file, 'r') as f:
        theme = json.load(f)
        print(f"✓ Loaded theme: {theme.get('name', theme_name)}")
        if 'description' in theme:
            print(f"  {theme['description']}")
        return theme

# Global theme variable - only used when calling create_poster() without a theme argument.
# The CLI and recommended usage pass themes directly to render_poster().
THEME = None

def create_gradient_fade(ax, color, location='bottom', zorder=10):
    """
    Creates a fade effect at the top or bottom of the map.
    """
    vals = np.linspace(0, 1, 256).reshape(-1, 1)
    gradient = np.hstack((vals, vals))
    
    rgb = mcolors.to_rgb(color)
    my_colors = np.zeros((256, 4))
    my_colors[:, 0] = rgb[0]
    my_colors[:, 1] = rgb[1]
    my_colors[:, 2] = rgb[2]
    
    if location == 'bottom':
        my_colors[:, 3] = np.linspace(1, 0, 256)
        extent_y_start = 0
        extent_y_end = 0.25
    else:
        my_colors[:, 3] = np.linspace(0, 1, 256)
        extent_y_start = 0.75
        extent_y_end = 1.0

    custom_cmap = mcolors.ListedColormap(my_colors)
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]
    
    y_bottom = ylim[0] + y_range * extent_y_start
    y_top = ylim[0] + y_range * extent_y_end
    
    ax.imshow(gradient, extent=[xlim[0], xlim[1], y_bottom, y_top],
              aspect='auto', cmap=custom_cmap, zorder=zorder, origin='lower')

# Road type classification mapping
ROAD_CATEGORIES = {
    'motorway': ['motorway', 'motorway_link'],
    'primary': ['trunk', 'trunk_link', 'primary', 'primary_link'],
    'secondary': ['secondary', 'secondary_link'],
    'tertiary': ['tertiary', 'tertiary_link'],
    'residential': ['residential', 'living_street', 'unclassified'],
}

ROAD_WIDTHS = {
    'motorway': 1.2,
    'primary': 1.0,
    'secondary': 0.8,
    'tertiary': 0.6,
    'residential': 0.4,
    'default': 0.4,
}


def get_road_category(highway):
    """
    Classify a highway type into a road category.
    Returns one of: motorway, primary, secondary, tertiary, residential, or default.
    """
    for category, types in ROAD_CATEGORIES.items():
        if highway in types:
            return category
    return 'default'


def get_highway_type(data):
    """
    Extract normalized highway type from edge data.
    Handles both string and list values from OSM.
    """
    highway = data.get('highway', 'unclassified')
    if isinstance(highway, list):
        return highway[0] if highway else 'unclassified'
    return highway


def get_edge_colors_by_type(G, theme):
    """
    Assigns colors to edges based on road type hierarchy.
    Returns a list of colors corresponding to each edge in the graph.
    """
    edge_colors = []
    for u, v, data in G.edges(data=True):
        highway = get_highway_type(data)
        category = get_road_category(highway)
        theme_key = f'road_{category}'
        color = theme.get(theme_key, theme['road_default'])
        edge_colors.append(color)
    return edge_colors


def get_edge_widths_by_type(G):
    """
    Assigns line widths to edges based on road type.
    Major roads get thicker lines.
    """
    edge_widths = []
    for u, v, data in G.edges(data=True):
        highway = get_highway_type(data)
        category = get_road_category(highway)
        width = ROAD_WIDTHS.get(category, ROAD_WIDTHS['default'])
        edge_widths.append(width)
    return edge_widths

def get_coordinates(city, country):
    """
    Fetches coordinates for a given city and country using geopy.
    Includes rate limiting to be respectful to the geocoding service.
    """
    print("Looking up coordinates...")
    geolocator = Nominatim(user_agent="city_map_poster")
    
    # Add a small delay to respect Nominatim's usage policy
    time.sleep(1)
    
    location = geolocator.geocode(f"{city}, {country}")
    
    if location:
        print(f"✓ Found: {location.address}")
        print(f"✓ Coordinates: {location.latitude}, {location.longitude}")
        return (location.latitude, location.longitude)
    else:
        raise ValueError(f"Could not find coordinates for {city}, {country}")

def extract_clipped_linestrings(geom, clip_poly):
    """
    Extract LineStrings from a geometry, clipped to a bounding polygon.
    Handles both LineString and MultiLineString geometries.
    """
    if geom is None:
        return []

    result = []
    # Normalize to list of base geometries
    geoms_to_process = [geom] if geom.geom_type == 'LineString' else []
    if geom.geom_type == 'MultiLineString':
        geoms_to_process = list(geom.geoms)

    for g in geoms_to_process:
        clipped = g.intersection(clip_poly)
        if clipped.is_empty:
            continue
        if clipped.geom_type == 'LineString':
            result.append(clipped)
        elif clipped.geom_type == 'MultiLineString':
            result.extend(clipped.geoms)

    return result


def fetch_land_polygon(point, dist):
    """
    Fetch land polygon using OSM coastline data for accurate alignment with roads.

    For inland cities (no coastline found), returns the entire bounding box as land.
    Returns a GeoDataFrame with land geometry, or None on error.
    """
    lat, lon = point
    # Convert meters to approximate degrees
    lat_delta = dist / METERS_PER_DEGREE_LAT
    lon_delta = dist / (METERS_PER_DEGREE_LAT * np.cos(np.radians(lat)))

    bbox_poly = box(
        lon - lon_delta, lat - lat_delta,
        lon + lon_delta, lat + lat_delta
    )
    minx, miny, maxx, maxy = bbox_poly.bounds

    # Try OSM coastline first
    try:
        # Fetch larger area to ensure we get complete coastlines
        coastlines = ox.features_from_point(
            point,
            tags={'natural': 'coastline'},
            dist=int(dist * COASTLINE_BUFFER_MULT)
        )

        if coastlines is None or coastlines.empty:
            # No coastline - inland city, entire bbox is land
            return gpd.GeoDataFrame({'geometry': [bbox_poly]}, crs='EPSG:4326')

        # Extract and clip linestrings
        lines = []
        for geom in coastlines.geometry:
            lines.extend(extract_clipped_linestrings(geom, bbox_poly))

        if not lines:
            return gpd.GeoDataFrame({'geometry': [bbox_poly]}, crs='EPSG:4326')

        # Build a complete set of lines: coastlines + bbox edges
        # Extend coastline endpoints to bbox edges
        bbox_edges = [
            LineString([(minx, miny), (maxx, miny)]),  # bottom
            LineString([(maxx, miny), (maxx, maxy)]),  # right
            LineString([(maxx, maxy), (minx, maxy)]),  # top
            LineString([(minx, maxy), (minx, miny)]),  # left
        ]

        # Combine all lines for polygonization
        all_lines = lines + bbox_edges
        combined = unary_union(all_lines)

        # Polygonize - this creates polygons from all enclosed areas
        polys, dangles, cuts, invalids = polygonize_full(combined)
        polygons = list(polys.geoms) if polys.geom_type == 'GeometryCollection' else [polys]

        if not polygons:
            raise Exception("Polygonization failed")

        # Filter to valid polygons within bbox
        valid_polys = []
        for p in polygons:
            if p.is_valid and not p.is_empty and p.area > 0:
                clipped = p.intersection(bbox_poly)
                if not clipped.is_empty and clipped.area > MIN_POLYGON_AREA:
                    valid_polys.append(clipped)

        if not valid_polys:
            raise Exception("No valid polygons after clipping")

        # Identify land vs sea
        # Strategy: find the polygon containing center (definitely land),
        # then identify the "sea" polygon as the largest one not containing center
        center = Point(lon, lat)

        # Find the main land polygon (contains center)
        main_land = None
        other_polys = []
        for poly in valid_polys:
            if poly.contains(center):
                main_land = poly
            else:
                other_polys.append(poly)

        if not main_land:
            raise Exception("No polygon contains center point")

        # Find the sea polygon - largest polygon that doesn't contain center
        # and has significant area (> 10% of bbox)
        sea_poly = None
        sea_area = 0
        for poly in other_polys:
            area_ratio = poly.area / bbox_poly.area
            if area_ratio > SEA_MIN_AREA_RATIO and poly.area > sea_area:
                # Check if this polygon borders the main land (shares coastline)
                # If it shares a long boundary with main_land, it's likely sea
                shared = poly.boundary.intersection(main_land.boundary).length
                if shared > SEA_MIN_SHARED_BOUNDARY:
                    sea_poly = poly
                    sea_area = poly.area

        # Classify remaining polygons as land or sea
        land_polys = [main_land]
        for poly in other_polys:
            if poly is sea_poly:
                continue  # Skip the identified sea polygon

            area_ratio = poly.area / bbox_poly.area
            # Small polygons are likely islands/land
            # Or if they don't share boundary with sea, they're land
            if area_ratio < ISLAND_MAX_AREA_RATIO:
                land_polys.append(poly)
            elif sea_poly is not None:
                shared_with_sea = poly.boundary.intersection(sea_poly.boundary).length
                shared_with_land = poly.boundary.intersection(main_land.boundary).length
                if shared_with_land > shared_with_sea:
                    land_polys.append(poly)

        if land_polys:
            land = unary_union(land_polys)
            return gpd.GeoDataFrame({'geometry': [land]}, crs='EPSG:4326')
        else:
            raise Exception("No land polygons identified")

    except Exception as e:
        print(f"  OSM coastline processing failed: {e}")
        return None

def fetch_map_data(point, dist, show_land=True):
    """
    Fetch all map data (streets, water, parks, land) from OpenStreetMap.
    Returns a dict with all the fetched data for rendering.
    """
    print("Fetching map data...")

    # Progress bar for data fetching
    total_steps = 4 if show_land else 3
    with tqdm(total=total_steps, desc="Fetching map data", unit="step", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
        # 1. Fetch Street Network
        pbar.set_description("Downloading street network")
        G = ox.graph_from_point(point, dist=dist, dist_type='bbox', network_type='all')
        pbar.update(1)
        time.sleep(0.5)  # Rate limit between requests

        # 2. Fetch Water Features
        pbar.set_description("Downloading water features")
        try:
            water = ox.features_from_point(
                point,
                tags={'natural': 'water', 'waterway': 'riverbank'},
                dist=dist
            )
        except Exception:
            water = None
        pbar.update(1)
        time.sleep(0.3)

        # 3. Fetch Parks
        pbar.set_description("Downloading parks/green spaces")
        try:
            parks = ox.features_from_point(point, tags={'leisure': 'park', 'landuse': 'grass'}, dist=dist)
        except Exception:
            parks = None
        pbar.update(1)

        # 4. Fetch Land Polygon (for sea/land distinction)
        if show_land:
            pbar.set_description("Processing land boundaries")
            try:
                land = fetch_land_polygon(point, dist)
            except Exception as e:
                print(f"Warning: Could not fetch land polygon: {e}")
                land = None
            pbar.update(1)
        else:
            land = None

    print("✓ All data downloaded successfully!")

    return {
        'G': G,
        'water': water,
        'parks': parks,
        'land': land
    }


def render_poster(city, country, point, map_data, theme, output_file, show_land=True):
    """
    Render a poster from pre-fetched map data using the specified theme.
    """
    G = map_data['G']
    water = map_data['water']
    parks = map_data['parks']
    land = map_data['land']

    # Setup Plot
    print("Rendering map...")
    if show_land:
        sea_color = theme.get('sea', theme['bg'])
    else:
        sea_color = theme['bg']  # Classic style - just background
    fig, ax = plt.subplots(figsize=(12, 16), facecolor=sea_color)
    ax.set_facecolor(sea_color)
    ax.set_position([0, 0, 1, 1])

    # Plot Layers

    # Layer 0.5: Land (drawn over sea background)
    if land is not None and not land.empty:
        land_color = theme.get('land', theme['bg'])
        land.plot(ax=ax, facecolor=land_color, edgecolor='none', zorder=0.5)

    # Layer 1: Water
    if water is not None and not water.empty:
        water.plot(ax=ax, facecolor=theme['water'], edgecolor='none', zorder=1)

    # Layer 2: Parks
    if parks is not None and not parks.empty:
        parks.plot(ax=ax, facecolor=theme['parks'], edgecolor='none', zorder=2)

    # Layer 3: Roads with hierarchy coloring
    print("Applying road hierarchy colors...")
    edge_colors = get_edge_colors_by_type(G, theme)
    edge_widths = get_edge_widths_by_type(G)

    ox.plot_graph(
        G, ax=ax, bgcolor=sea_color,
        node_size=0,
        edge_color=edge_colors,
        edge_linewidth=edge_widths,
        show=False, close=False
    )

    # Layer 4: Gradients (Top and Bottom)
    create_gradient_fade(ax, theme['gradient_color'], location='bottom', zorder=10)
    create_gradient_fade(ax, theme['gradient_color'], location='top', zorder=10)

    # Typography using Roboto font (fallback to system monospace)
    if FONTS:
        font_main = FontProperties(fname=FONTS['bold'], size=60)
        font_sub = FontProperties(fname=FONTS['light'], size=22)
        font_coords = FontProperties(fname=FONTS['regular'], size=14)
        font_attr = FontProperties(fname=FONTS['light'], size=8)
    else:
        font_main = FontProperties(family='monospace', weight='bold', size=60)
        font_sub = FontProperties(family='monospace', weight='normal', size=22)
        font_coords = FontProperties(family='monospace', size=14)
        font_attr = FontProperties(family='monospace', size=8)

    spaced_city = "  ".join(list(city.upper()))

    # --- BOTTOM TEXT ---
    ax.text(0.5, 0.14, spaced_city, transform=ax.transAxes,
            color=theme['text'], ha='center', fontproperties=font_main, zorder=11)

    ax.text(0.5, 0.10, country.upper(), transform=ax.transAxes,
            color=theme['text'], ha='center', fontproperties=font_sub, zorder=11)

    lat, lon = point
    lat_dir = "N" if lat >= 0 else "S"
    lon_dir = "E" if lon >= 0 else "W"
    coords = f"{abs(lat):.4f}° {lat_dir} / {abs(lon):.4f}° {lon_dir}"

    ax.text(0.5, 0.07, coords, transform=ax.transAxes,
            color=theme['text'], alpha=0.7, ha='center', fontproperties=font_coords, zorder=11)

    ax.plot([0.4, 0.6], [0.125, 0.125], transform=ax.transAxes,
            color=theme['text'], linewidth=1, zorder=11)

    # --- ATTRIBUTION (bottom right) ---
    ax.text(0.98, 0.02, "© OpenStreetMap contributors", transform=ax.transAxes,
            color=theme['text'], alpha=0.5, ha='right', va='bottom',
            fontproperties=font_attr, zorder=11)

    # Save
    print(f"Saving to {output_file}...")
    plt.savefig(output_file, dpi=300, facecolor=sea_color)
    plt.close()
    print(f"✓ Done! Poster saved as {output_file}")


def create_poster(city, country, point, dist, output_file, theme=None, show_land=True):
    """
    Convenience function that fetches map data and renders a poster in one call.

    For multi-theme generation, use fetch_map_data() + render_poster() directly
    to avoid redundant data fetching.

    Args:
        theme: Optional theme dict. If None, falls back to global THEME variable.
    """
    if theme is None:
        if THEME is None:
            raise RuntimeError("THEME not loaded. Call load_theme() or pass a theme dict.")
        theme = THEME

    print(f"\nGenerating map for {city}, {country}...")
    map_data = fetch_map_data(point, dist, show_land)
    render_poster(city, country, point, map_data, theme, output_file, show_land)


def print_examples():
    """Print usage examples."""
    print("""
City Map Poster Generator
=========================

Usage:
  python create_map_poster.py --city <city> --country <country> [options]

Examples:
  # Iconic grid patterns
  python create_map_poster.py -c "New York" -C "USA" -t noir -d 12000           # Manhattan grid
  python create_map_poster.py -c "Barcelona" -C "Spain" -t warm_beige -d 8000   # Eixample district grid
  
  # Waterfront & canals
  python create_map_poster.py -c "Venice" -C "Italy" -t blueprint -d 4000       # Canal network
  python create_map_poster.py -c "Amsterdam" -C "Netherlands" -t ocean -d 6000  # Concentric canals
  python create_map_poster.py -c "Dubai" -C "UAE" -t midnight_blue -d 15000     # Palm & coastline
  
  # Radial patterns
  python create_map_poster.py -c "Paris" -C "France" -t pastel_dream -d 10000   # Haussmann boulevards
  python create_map_poster.py -c "Moscow" -C "Russia" -t noir -d 12000          # Ring roads
  
  # Organic old cities
  python create_map_poster.py -c "Tokyo" -C "Japan" -t japanese_ink -d 15000    # Dense organic streets
  python create_map_poster.py -c "Marrakech" -C "Morocco" -t terracotta -d 5000 # Medina maze
  python create_map_poster.py -c "Rome" -C "Italy" -t warm_beige -d 8000        # Ancient street layout
  
  # Coastal cities
  python create_map_poster.py -c "San Francisco" -C "USA" -t sunset -d 10000    # Peninsula grid
  python create_map_poster.py -c "Sydney" -C "Australia" -t ocean -d 12000      # Harbor city
  python create_map_poster.py -c "Mumbai" -C "India" -t contrast_zones -d 18000 # Coastal peninsula
  
  # River cities
  python create_map_poster.py -c "London" -C "UK" -t noir -d 15000              # Thames curves
  python create_map_poster.py -c "Budapest" -C "Hungary" -t copper_patina -d 8000  # Danube split

  # Multiple themes (fetches data once, renders multiple posters)
  python create_map_poster.py -c "Paris" -C "France" -t noir midnight_blue sunset -d 10000

  # List themes
  python create_map_poster.py --list-themes

Options:
  --city, -c        City name (required)
  --country, -C     Country name (required)
  --theme, -t       Theme name(s) - can specify multiple (default: feature_based)
  --distance, -d    Map radius in meters (default: 29000)
  --format, -f      Output format: png (raster) or svg (vector) (default: png)
  --no-land         Disable land/sea polygons (classic style, faster)
  --list-themes     List all available themes

Distance guide:
  4000-6000m   Small/dense cities (Venice, Amsterdam old center)
  8000-12000m  Medium cities, focused downtown (Paris, Barcelona)
  15000-20000m Large metros, full city view (Tokyo, Mumbai)

Available themes can be found in the 'themes/' directory.
Generated posters are saved to 'posters/' directory.
""")

def list_themes():
    """List all available themes with descriptions."""
    available_themes = get_available_themes()
    if not available_themes:
        print("No themes found in 'themes/' directory.")
        return
    
    print("\nAvailable Themes:")
    print("-" * 60)
    for theme_name in available_themes:
        theme_path = os.path.join(THEMES_DIR, f"{theme_name}.json")
        try:
            with open(theme_path, 'r') as f:
                theme_data = json.load(f)
                display_name = theme_data.get('name', theme_name)
                description = theme_data.get('description', '')
        except Exception:
            display_name = theme_name
            description = ''
        print(f"  {theme_name}")
        print(f"    {display_name}")
        if description:
            print(f"    {description}")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate beautiful map posters for any city",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_map_poster.py --city "New York" --country "USA"
  python create_map_poster.py --city Tokyo --country Japan --theme midnight_blue
  python create_map_poster.py --city Paris --country France --theme noir --distance 15000
  python create_map_poster.py --city Paris --country France -t noir midnight_blue sunset  # Multiple themes
  python create_map_poster.py --list-themes
        """
    )
    
    parser.add_argument('--city', '-c', type=str, help='City name')
    parser.add_argument('--country', '-C', type=str, help='Country name')
    parser.add_argument('--theme', '-t', type=str, nargs='+', default=['feature_based'],
                        help='Theme name(s) - can specify multiple (default: feature_based)')
    parser.add_argument('--distance', '-d', type=int, default=29000, help='Map radius in meters (default: 29000)')
    parser.add_argument('--format', '-f', default='png', choices=['png', 'svg'], help='Output format: png (raster) or svg (vector)')
    parser.add_argument('--no-land', action='store_true', help='Disable land/sea polygons (classic style, faster)')
    parser.add_argument('--list-themes', action='store_true', help='List all available themes')
    
    args = parser.parse_args()
    
    # If no arguments provided, show examples
    if len(sys.argv) == 1:
        print_examples()
        sys.exit(0)
    
    # List themes if requested
    if args.list_themes:
        list_themes()
        sys.exit(0)
    
    # Validate required arguments
    if not args.city or not args.country:
        print("Error: --city and --country are required.\n")
        print_examples()
        sys.exit(1)
    
    # Validate all themes exist
    available_themes = get_available_themes()
    invalid_themes = [t for t in args.theme if t not in available_themes]
    if invalid_themes:
        print(f"Error: Theme(s) not found: {', '.join(invalid_themes)}")
        print(f"Available themes: {', '.join(available_themes)}")
        sys.exit(1)

    print("=" * 50)
    print("City Map Poster Generator")
    print("=" * 50)

    # Get coordinates and generate poster(s)
    try:
        coords = get_coordinates(args.city, args.country)

        # Fetch map data once
        print(f"\nGenerating map for {args.city}, {args.country}...")
        show_land = not args.no_land
        map_data = fetch_map_data(coords, args.distance, show_land)

        # Render poster for each theme
        for theme_name in args.theme:
            print(f"\n--- Rendering with theme: {theme_name} ---")
            theme = load_theme(theme_name)
            output_file = generate_output_filename(args.city, theme_name, args.format)
            render_poster(args.city, args.country, coords, map_data, theme, output_file,
                          show_land=show_land)

        print("\n" + "=" * 50)
        print(f"✓ Poster generation complete! ({len(args.theme)} poster(s) created)")
        print("=" * 50)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
