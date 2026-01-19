# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

City Map Poster Generator - a Python tool that creates minimalist map posters for any city using OpenStreetMap data.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Generate a poster
python create_map_poster.py --city <city> --country <country> [--theme <theme>] [--distance <meters>] [--format png|svg]

# List available themes
python create_map_poster.py --list-themes
```

## Architecture

Single-file application (`create_map_poster.py`) with this pipeline:

```
CLI (argparse) → Geocoding (Nominatim/geopy) → Data Fetch (OSMnx) → Render (matplotlib) → PNG/SVG Output
```

**Rendering layers (z-order):**
- z=0: Background color
- z=1: Water polygons
- z=2: Parks polygons
- z=3: Street network (via `ox.plot_graph`)
- z=10: Gradient fades (top/bottom)
- z=11: Text labels

**Key functions:**
- `get_coordinates()` - Geocodes city/country via Nominatim
- `create_poster()` - Main pipeline: fetches OSM data, renders layers, saves output
- `get_edge_colors_by_type()` / `get_edge_widths_by_type()` - Maps OSM highway tags to styling
- `load_theme()` - Loads JSON theme with fallback defaults

## Theme System

Themes are JSON files in `themes/` directory. Required properties:
- `bg`, `text`, `gradient_color`, `water`, `parks`
- Road colors: `road_motorway`, `road_primary`, `road_secondary`, `road_tertiary`, `road_residential`, `road_default`

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
