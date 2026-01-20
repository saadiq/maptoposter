# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

City Map Poster Generator - a Python tool that creates minimalist map posters for any city using OpenStreetMap data.

## Commands

This project uses `uv` for dependency management and running Python.

```bash
# Install dependencies
uv sync

# Generate a poster (single theme)
uv run python create_map_poster.py --city <city> --country <country> [--theme <theme>] [--distance <meters>] [--format png|svg] [--no-land]

# Generate with custom coordinates (skip geocoding)
uv run python create_map_poster.py --city <city> --country <country> --lat <latitude> --lon <longitude> [--theme <theme>]

# Generate multiple posters (fetches data once, renders with each theme)
uv run python create_map_poster.py --city <city> --country <country> -t theme1 theme2 theme3 [--distance <meters>]

# List available themes
uv run python create_map_poster.py --list-themes
```

## Architecture

Single-file application (`create_map_poster.py`) with this pipeline:

```
CLI (argparse) → Geocoding (Nominatim/geopy, or --lat/--lon override) → Data Fetch (OSMnx) → Render (matplotlib) → PNG/SVG Output
```

**Multi-theme support:** Data is fetched once via `fetch_map_data()`, then `render_poster()` is called for each theme. This avoids redundant API calls when generating multiple poster variants.

**Rendering layers (z-order):**
- z=0: Sea color (background)
- z=0.5: Land polygons (OSM coastline data)
- z=1: Water polygons (rivers, lakes from OSM)
- z=2: Parks polygons
- z=3: Street network (via `ox.plot_graph`)
- z=10: Gradient fades (top/bottom)
- z=11: Text labels

**Key functions:**
- `get_coordinates()` - Geocodes city/country via Nominatim
- `fetch_land_polygon()` - Fetches land boundaries from OSM coastline data
- `fetch_map_data(point, dist, show_land)` - Fetches all OSM data (streets, water, parks, land), returns dict
- `render_poster(city, country, point, map_data, theme, output_file, show_land)` - Renders poster from pre-fetched data
- `create_poster(city, country, point, dist, output_file, theme, show_land)` - Convenience wrapper that fetches and renders in one call
- `get_edge_colors_by_type(G, theme)` / `get_edge_widths_by_type(G)` - Maps OSM highway tags to styling
- `load_theme()` - Loads JSON theme with fallback defaults

## Theme System

Themes are JSON files in `themes/` directory. Required properties:
- `bg`, `text`, `gradient_color`, `water`, `parks`, `sea`, `land`
- Road colors: `road_motorway`, `road_primary`, `road_secondary`, `road_tertiary`, `road_residential`, `road_default`

For coastal cities, `sea` defines ocean color and `land` defines landmass color. For inland cities, land covers the entire map. Themes without `sea`/`land` fall back to `bg` color.

## Adding Map Layers

Use OSMnx patterns:
```python
# Fetch features
features = ox.features_from_point(point, tags={'railway': 'rail'}, dist=dist)
# Plot with z-order between existing layers
features.plot(ax=ax, color=THEME['new_property'], linewidth=0.5, zorder=2.5)
```

## Notes

- Nominatim has rate limits - code includes delays
- Large distances (>20km) are slow and memory-intensive
- Output goes to `posters/` as `{city}_{theme}_{timestamp}.{png|svg}` (PNG at 300 DPI, SVG vector)
