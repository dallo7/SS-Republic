# -*- coding: utf-8 -*-
from datetime import datetime, date, timezone, timedelta
import pytz
import numpy as np
import pandas as pd
import dash
from dash import dcc, html, callback, Input, Output, State, dash_table, no_update, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import requests
import math
import traceback
import csv
from io import StringIO
import base64
import dash_auth
import flight_client

# --- Constants ---
API_BASE_URL = "http://13.239.238.138:5070"
FLIGHT_ID_COLUMN = "flight_id"
DISPLAY_TIMEZONE = pytz.timezone('Africa/Nairobi')
PROCESSING_TIMEZONE = pytz.utc
CHART_FONT_COLOR = "#adb5bd"
CHART_PAPER_BG = 'rgba(0,0,0,0)'
CHART_PLOT_BG = 'rgba(0,0,0,0)'
LOGO_FILENAME = "image.png"

# Visualizer Tool Constants
DEFAULT_MAP_STYLE_VIZ = "carto-darkmatter"
FLIGHT_PATH_COLOR_VIZ = 'lime'
SOUTH_SUDAN_AIRSPACE_LINE_COLOR_VIZ = 'yellow'
FINAL_DESTINATION_POINT_COLOR_VIZ = 'blue'
SOUTH_SUDAN_AIRSPACE_TRACE_NAME_VIZ = "South Sudan Airspace"
FINAL_DATA_POINT_TRACE_NAME_VIZ = "Final Data Point"
CALLSIGN_VISUALIZER_API_ENDPOINT = f"{API_BASE_URL}/api/flights/by_callsign"
CALLSIGN_BATCH_SIZE = 1000
JUB_SOUTH_SUDAN_MIN_LAT_VIZ = 3.0
JUB_SOUTH_SUDAN_MAX_LAT_VIZ = 13.0
JUB_SOUTH_SUDAN_MIN_LON_VIZ = 24.0
JUB_SOUTH_SUDAN_MAX_LON_VIZ = 36.0
SOUTH_SUDAN_CENTER_LAT_VIZ = (JUB_SOUTH_SUDAN_MIN_LAT_VIZ + JUB_SOUTH_SUDAN_MAX_LAT_VIZ) / 2
SOUTH_SUDAN_CENTER_LON_VIZ = (JUB_SOUTH_SUDAN_MIN_LON_VIZ + JUB_SOUTH_SUDAN_MAX_LON_VIZ) / 2
SOUTH_SUDAN_MAP_ZOOM_VIZ = 4.5
INITIAL_TABLE_COLUMNS_VIZ = [{"name": "Record ID", "id": FLIGHT_ID_COLUMN}, {"name": "Callsign", "id": "callsign"},
                             {"name": "Last Update Time (EAT)", "id": "LAST_UPDATE_TIME_HOVER"},
                             {"name": "Latitude", "id": "LATITUDE"}, {"name": "Longitude", "id": "LONGITUDE"},
                             {"name": "Altitude", "id": "ALTITUDE_display"}, {"name": "Speed", "id": "SPEED_display"},
                             {"name": "Track", "id": "TRACK_display"}]
CUSTOM_DATA_COLUMNS_VIZ = [FLIGHT_ID_COLUMN, 'ALTITUDE_display', 'LAST_UPDATE_TIME_HOVER', 'SPEED_display',
                           'TRACK_display', 'callsign']
HOVER_TEMPLATE_FLIGHT_POINT_VIZ = "<b>Callsign:</b> %{customdata[5]}<br><b>Rec ID:</b> %{customdata[0]}<br><b>Lat:</b> %{lat:.4f}, <b>Lon:</b> %{lon:.4f}<br><b>Alt:</b> %{customdata[1]} ft<br><b>Time:</b> %{customdata[2]}<br><b>Speed:</b> %{customdata[3]} kts, <b>Track:</b> %{customdata[4]}°<extra></extra>"

# Investigation Tool Constants
FETCH_LIMIT_INV = 50000
FETCH_INTERVAL_SECONDS_INV = 60
ON_MAP_TEXT_SIZE_INV = 9
NA_REPLACE_VALUES_INV = ['', 'nan', 'NaN', 'None', 'null', 'NONE', 'NULL', '#N/A', 'N/A', 'NA', '-']
JUB_SOUTH_SUDAN_MIN_LAT_INV = 3.0;
JUB_SOUTH_SUDAN_MAX_LAT_INV = 13.0;
JUB_SOUTH_SUDAN_MIN_LON_INV = 24.0;
JUB_SOUTH_SUDAN_MAX_LON_INV = 36.0
JUB_SPECIFIC_MIN_LAT_INV = 4.96;
JUB_SPECIFIC_MAX_LAT_INV = 10.76
JUB_SPECIFIC_MIN_LON_INV = 26.76
JUB_SPECIFIC_MAX_LON_INV = 32.62
JUB_SPECIFIC_CENTER_LAT_INV = (JUB_SPECIFIC_MIN_LAT_INV + JUB_SPECIFIC_MAX_LAT_INV) / 2
JUB_SPECIFIC_CENTER_LON_INV = (JUB_SPECIFIC_MIN_LON_INV + JUB_SPECIFIC_MAX_LON_INV) / 2
JUB_BILLING_MAP_STYLE_INV = "carto-positron"
AIRLINE_DATA_URL_INV = 'https://raw.githubusercontent.com/jpatokal/openflights/master/data/airlines.dat'
iata_to_airline_map = {}
icao_to_airline_map = {}


def get_logo_base64():
    try:
        with open(f"assets/{LOGO_FILENAME}", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        return f"data:image/png;base64,{encoded_string}"
    except FileNotFoundError:
        print(f"ERROR: Logo file '{LOGO_FILENAME}' not found in 'assets' folder.");
        return None
    except Exception as e:
        print(f"Error encoding logo: {e}");
        return None


APP_LOGO_B64 = get_logo_base64()


def build_airline_dicts():
    global iata_to_airline_map, icao_to_airline_map;
    temp_iata_map, temp_icao_map = {}, {}
    print(f"INFO: [{datetime.now(timezone.utc).isoformat(timespec='seconds')}] Fetching airline data...")
    try:
        resp = requests.get(AIRLINE_DATA_URL_INV, timeout=15);
        resp.raise_for_status()
        reader = csv.reader(StringIO(resp.text))
        for row in reader:
            if len(row) < 7: continue
            name, iata, icao, country = row[1].strip(), row[3].strip().upper(), row[4].strip().upper(), row[6].strip()
            name = "N/A" if not name or name == "\\N" else name;
            country = "N/A" if not country or country == "\\N" else country
            if name != "N/A":
                if iata and iata != "\\N" and len(iata) == 2: temp_iata_map[iata] = (name, country)
                if icao and icao != "\\N" and len(icao) == 3: temp_icao_map[icao] = (name, country)
        iata_to_airline_map, icao_to_airline_map = temp_iata_map, temp_icao_map
        print(f"INFO: Loaded {len(iata_to_airline_map)} IATA and {len(icao_to_airline_map)} ICAO codes.")
    except Exception as e:
        print(f"CRITICAL ERROR: Airline data fetch/parse failed: {e}");
        traceback.print_exc()


build_airline_dicts()

fetcher_inv = None
if flight_client and hasattr(flight_client, 'FlightDataFetcher'):
    try:
        fetcher_inv = flight_client.FlightDataFetcher(base_url=API_BASE_URL, api_endpoint="/api/flights/latest_unique",
                                                      fetch_limit=FETCH_LIMIT_INV)
    except Exception as e:
        print(f"ERROR: FlightDataFetcher init failed for Investigation Tool: {e}")
else:
    print("ERROR: flight_client.FlightDataFetcher not available. Investigation Tool live data will be unavailable.")

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.VAPOR, dbc.icons.FONT_AWESOME],
                suppress_callback_exceptions=True, title="Flight Operations Dashboard",
                prevent_initial_callbacks=True)  # Default prevent_initial_call for all callbacks
app.secret_key = 'your_very_secret_key_for_unified_flight_dashboard_v17'

server = app.server

auth_handler = None
if dash_auth and hasattr(dash_auth, 'BasicAuth') and hasattr(dash_auth, 'VALID_USERNAME_PASSWORD_PAIRS'):
    try:
        auth_handler = dash_auth.BasicAuth(app, dash_auth.VALID_USERNAME_PASSWORD_PAIRS)
    except Exception as auth_e:
        print(f"ERROR setting up authentication: {auth_e}.")
else:
    print("WARNING: dash_auth module, BasicAuth class, or credentials not found. Auth disabled.")


def get_font_color_for_dark_map_viz(): return CHART_FONT_COLOR


def is_in_south_sudan_airspace_viz(lat, lon):
    if pd.isna(lat) or pd.isna(lon): return False
    return (JUB_SOUTH_SUDAN_MIN_LAT_VIZ <= lat <= JUB_SOUTH_SUDAN_MAX_LAT_VIZ and
            JUB_SOUTH_SUDAN_MIN_LON_VIZ <= lon <= JUB_SOUTH_SUDAN_MAX_LON_VIZ)


def create_flight_scatter_trace_viz(lon_series, lat_series, alt_series_for_size_calc, custom_data_df,
                                    trace_name, mode="lines+markers",
                                    line_color=FLIGHT_PATH_COLOR_VIZ, line_width=2,
                                    marker_color=FLIGHT_PATH_COLOR_VIZ, marker_base_size_calc=True,
                                    hover_template=None, hoverinfo=None, opacity=1.0,
                                    custom_marker_config=None, meta=None):
    marker_dict = {'opacity': opacity}
    if custom_marker_config:
        marker_dict.update(custom_marker_config)
        if 'color' not in marker_dict: marker_dict['color'] = marker_color
    elif marker_base_size_calc and "markers" in mode:
        alt_numeric = pd.to_numeric(alt_series_for_size_calc, errors='coerce').fillna(30000)
        marker_dict['size'] = (alt_numeric / 5000 + 4).clip(4, 12);
        marker_dict['color'] = marker_color
    elif "markers" in mode:
        marker_dict['size'] = 6;
        marker_dict['color'] = marker_color
    trace = go.Scattermapbox(mode=mode, lon=lon_series, lat=lat_series,
                             marker=marker_dict if "markers" in mode else None,
                             line=dict(width=line_width, color=line_color) if "lines" in mode else None,
                             name=trace_name, customdata=custom_data_df, hovertemplate=hover_template, meta=meta,
                             opacity=opacity)
    if hoverinfo: trace.hoverinfo = hoverinfo
    return trace


def update_figure_layout_viz(fig, title_text, center_lat, center_lon, zoom_level,
                             map_style=DEFAULT_MAP_STYLE_VIZ, font_color=CHART_FONT_COLOR,
                             paper_bg=CHART_PAPER_BG, plot_bg=CHART_PLOT_BG, height=550,
                             showlegend=True, uirevision_key="default_uirevision"):
    fig.update_layout(mapbox_style=map_style, margin={"r": 5, "t": 35, "l": 5, "b": 5},
                      title=dict(text=title_text, x=0.5, font=dict(color=font_color, size=16)),
                      font=dict(color=font_color), paper_bgcolor=paper_bg, plot_bgcolor=plot_bg,
                      mapbox_center={"lat": center_lat, "lon": center_lon}, mapbox_zoom=zoom_level, height=height,
                      showlegend=showlegend,
                      legend=dict(font=dict(color=font_color), bgcolor='rgba(0,0,0,0.5)', bordercolor='grey',
                                  borderwidth=1), uirevision=uirevision_key)


def create_empty_map_figure_dict_viz(message="No data available",
                                     center_lat=SOUTH_SUDAN_CENTER_LAT_VIZ, center_lon=SOUTH_SUDAN_CENTER_LON_VIZ,
                                     zoom=SOUTH_SUDAN_MAP_ZOOM_VIZ):
    title_color = get_font_color_for_dark_map_viz()
    layout = go.Layout(
        mapbox=dict(style=DEFAULT_MAP_STYLE_VIZ, center={"lat": center_lat, "lon": center_lon}, zoom=zoom),
        margin=dict(r=5, t=25, l=5, b=5), paper_bgcolor=CHART_PAPER_BG, plot_bgcolor=CHART_PLOT_BG,
        font=dict(color=title_color), title=dict(text=message, x=0.5, font=dict(size=16, color=title_color)))
    return go.Figure(data=[], layout=layout).to_dict()


def fetch_one_page_by_callsign_viz(callsign_to_fetch, page_num, records_per_page, api_endpoint):
    params = {'callsign': callsign_to_fetch, 'page': page_num, 'limit': records_per_page}
    response_data, error_message_detail = None, None
    try:
        response_obj = requests.get(api_endpoint, params=params, timeout=20)
        response_obj.raise_for_status()
        response_data = response_obj.json()
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response else 'N/A';
        error_message_detail = f"API HTTP Error ({status_code}) on page {page_num}."
    except requests.exceptions.Timeout:
        error_message_detail = f"API request timed out on page {page_num}."
    except requests.exceptions.RequestException as req_err:
        error_message_detail = f"Network/Request error: {req_err}."
    except ValueError:
        error_message_detail = f"Invalid data format from API (not JSON)."
    return response_data, error_message_detail


def normalize_flight_data_df_viz(df_flights, callsign_for_title="N/A"):
    if not isinstance(df_flights, pd.DataFrame) or df_flights.empty: return pd.DataFrame()
    df = df_flights.copy();
    temp_id_col = '_original_flight_id_holder_'
    rename_map = {'lat': 'LATITUDE', 'lon': 'LONGITUDE', 'alt': 'ALTITUDE', 'last_update': 'LAST_UPDATE_TIME',
                  'speed': 'SPEED', 'track': 'TRACK'}
    if 'flight_id' in df.columns: rename_map['flight_id'] = temp_id_col
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
    if 'LATITUDE' not in df.columns or 'LONGITUDE' not in df.columns: return pd.DataFrame()
    for col in ['LATITUDE', 'LONGITUDE', 'ALTITUDE', 'SPEED', 'TRACK']:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['LATITUDE', 'LONGITUDE'], inplace=True);
    if df.empty: return df
    if 'LAST_UPDATE_TIME' in df.columns:
        df['LAST_UPDATE_TIME'] = pd.to_datetime(df['LAST_UPDATE_TIME'], errors='coerce', utc=True);
        df['DATE_ONLY'] = df['LAST_UPDATE_TIME'].dt.date
    else:
        df['LAST_UPDATE_TIME'] = pd.NaT;
        df['DATE_ONLY'] = pd.NaT
    df.sort_values(by='LAST_UPDATE_TIME', ascending=True, inplace=True, na_position='first');
    df.reset_index(drop=True, inplace=True)
    idx_ids = "trk_" + pd.Series(df.index).astype(str)
    if temp_id_col in df.columns:
        df[FLIGHT_ID_COLUMN] = df[temp_id_col].fillna(idx_ids);
        df.drop(columns=[temp_id_col], inplace=True)
    elif FLIGHT_ID_COLUMN not in df.columns:
        df[FLIGHT_ID_COLUMN] = idx_ids
    else:
        df[FLIGHT_ID_COLUMN] = df[FLIGHT_ID_COLUMN].fillna(idx_ids)
    df[FLIGHT_ID_COLUMN] = df[FLIGHT_ID_COLUMN].astype(str)
    if pd.api.types.is_datetime64_any_dtype(df['LAST_UPDATE_TIME']) and not df['LAST_UPDATE_TIME'].isnull().all():
        if df['LAST_UPDATE_TIME'].dt.tz is None: df['LAST_UPDATE_TIME'] = df['LAST_UPDATE_TIME'].dt.tz_localize('UTC')
        df['LAST_UPDATE_TIME_SLIDER_MARK'] = df['LAST_UPDATE_TIME'].dt.tz_convert(DISPLAY_TIMEZONE).dt.strftime('%H:%M')
        df['LAST_UPDATE_TIME_HOVER'] = df['LAST_UPDATE_TIME'].dt.tz_convert(DISPLAY_TIMEZONE).dt.strftime(
            '%Y-%m-%d %H:%M:%S %Z')
    else:
        df['LAST_UPDATE_TIME_SLIDER_MARK'] = df.index.astype(str);
        df['LAST_UPDATE_TIME_HOVER'] = 'N/A'
    for col_name in ['ALTITUDE', 'SPEED', 'TRACK']:
        disp_col = f'{col_name}_display'
        if col_name in df.columns:
            df[disp_col] = df[col_name].fillna('N/A').astype(str)
        else:
            df[disp_col] = 'N/A'
    if 'callsign' not in df.columns:
        df['callsign'] = callsign_for_title
    else:
        df['callsign'] = df['callsign'].fillna(callsign_for_title)
    return df


def generate_table_parts_viz(df_normalized):
    if not isinstance(df_normalized, pd.DataFrame) or df_normalized.empty: return [], INITIAL_TABLE_COLUMNS_VIZ
    table_ids = [c["id"] for c in INITIAL_TABLE_COLUMNS_VIZ];
    actual_cols = [c for c in table_ids if c in df_normalized.columns]
    if not actual_cols: return [], INITIAL_TABLE_COLUMNS_VIZ
    df_view = df_normalized[actual_cols].copy()
    if 'LATITUDE' in df_view: df_view['LATITUDE'] = df_view['LATITUDE'].round(4)
    if 'LONGITUDE' in df_view: df_view['LONGITUDE'] = df_view['LONGITUDE'].round(4)
    tbl_data = df_view.to_dict('records')
    curr_cols = [c for c in INITIAL_TABLE_COLUMNS_VIZ if c["id"] in df_view.columns]
    if not curr_cols and df_view.columns.any():
        curr_cols = [{"name": c.replace('_HOVER', ' (EAT)').replace('_display', '').replace(FLIGHT_ID_COLUMN,
                                                                                            'Record ID').replace('_',
                                                                                                                 ' ').title(),
                      "id": c} for c in df_view.columns]
    elif not curr_cols:
        curr_cols = INITIAL_TABLE_COLUMNS_VIZ
    return tbl_data, curr_cols


def update_map_and_table_for_display_viz(records_df, callsign, map_title_suffix=""):
    if not isinstance(records_df, pd.DataFrame) or records_df.empty: empty_fig = create_empty_map_figure_dict_viz(
        f"No data for {callsign}. {map_title_suffix}"); return empty_fig, [], INITIAL_TABLE_COLUMNS_VIZ, None
    df_display = records_df
    if df_display.empty or 'LATITUDE' not in df_display.columns or 'LONGITUDE' not in df_display.columns: empty_fig = create_empty_map_figure_dict_viz(
        f"No valid coords for {callsign}. {map_title_suffix}"); return empty_fig, [], INITIAL_TABLE_COLUMNS_VIZ, None
    fig = go.Figure()
    custom_cols = [c for c in CUSTOM_DATA_COLUMNS_VIZ if c in df_display.columns]
    main_trace = create_flight_scatter_trace_viz(df_display["LONGITUDE"], df_display["LATITUDE"],
                                                 df_display['ALTITUDE'],
                                                 df_display[custom_cols] if custom_cols else None,
                                                 callsign if callsign else "Flight Path", mode="lines",
                                                 line_color=FLIGHT_PATH_COLOR_VIZ,
                                                 hover_template=HOVER_TEMPLATE_FLIGHT_POINT_VIZ if custom_cols else None,
                                                 hoverinfo='text' if custom_cols else None)
    fig.add_trace(main_trace)
    df_ss = df_display[
        df_display.apply(lambda r: is_in_south_sudan_airspace_viz(r['LATITUDE'], r['LONGITUDE']), axis=1)]
    if not df_ss.empty:
        ss_data = df_ss[custom_cols] if custom_cols else None
        ss_trace = create_flight_scatter_trace_viz(df_ss["LONGITUDE"], df_ss["LATITUDE"], df_ss['ALTITUDE'], ss_data,
                                                   SOUTH_SUDAN_AIRSPACE_TRACE_NAME_VIZ, mode="lines",
                                                   line_color=SOUTH_SUDAN_AIRSPACE_LINE_COLOR_VIZ,
                                                   hover_template=HOVER_TEMPLATE_FLIGHT_POINT_VIZ if custom_cols else None,
                                                   hoverinfo='text' if custom_cols else None)
        fig.add_trace(ss_trace)
    if not df_display.empty:
        final_pt = df_display.iloc[[-1]];
        final_data = final_pt[custom_cols] if custom_cols else None
        final_trace = create_flight_scatter_trace_viz(final_pt["LONGITUDE"], final_pt["LATITUDE"], final_pt['ALTITUDE'],
                                                      final_data, FINAL_DATA_POINT_TRACE_NAME_VIZ, mode="markers",
                                                      custom_marker_config={'size': 10,
                                                                            'color': FINAL_DESTINATION_POINT_COLOR_VIZ,
                                                                            'symbol': 'circle'},
                                                      marker_base_size_calc=False,
                                                      hover_template=HOVER_TEMPLATE_FLIGHT_POINT_VIZ if custom_cols else None,
                                                      hoverinfo='text' if custom_cols else None)
        fig.add_trace(final_trace)
    center_lat, center_lon = (df_display["LATITUDE"].mean(),
                              df_display["LONGITUDE"].mean()) if not df_display.empty else (SOUTH_SUDAN_CENTER_LAT_VIZ,
                                                                                            SOUTH_SUDAN_CENTER_LON_VIZ)
    update_figure_layout_viz(fig, f"Path for {callsign} ({len(df_display)} pts) - {map_title_suffix}",
                             float(center_lat), float(center_lon), SOUTH_SUDAN_MAP_ZOOM_VIZ,
                             uirevision_key=f"{callsign}_{DEFAULT_MAP_STYLE_VIZ}_{map_title_suffix.replace(' ', '_')}")
    tbl_data, tbl_cols = generate_table_parts_viz(df_display)
    return fig.to_dict(), tbl_data, tbl_cols, df_display.to_dict('records')


def lookup_airline_info_by_code_inv(code):
    code = str(code).strip().upper();
    if not code: return None
    if len(code) == 3 and code in icao_to_airline_map:
        return icao_to_airline_map[code]
    elif len(code) == 2 and code in iata_to_airline_map:
        return iata_to_airline_map[code]
    elif code in icao_to_airline_map:
        return icao_to_airline_map[code]
    elif code in iata_to_airline_map:
        return iata_to_airline_map[code]
    return None


def get_airline_info_from_callsign_inv(cs_str):
    default = ('N/A', 'N/A')
    if not cs_str or pd.isna(cs_str) or str(cs_str).strip().upper() in NA_REPLACE_VALUES_INV + ['N/A', 'NONE', 'NULL',
                                                                                                '']: return default
    cs_str = str(cs_str).strip().upper()
    if not iata_to_airline_map and not icao_to_airline_map: return ('Airline Data Missing', 'Airline Data Missing')
    found_info = None
    if len(cs_str) >= 3:
        prefix = cs_str[:3]
        if prefix.isalpha():
            info = lookup_airline_info_by_code_inv(prefix)
            if info and info[0] != 'N/A': found_info = info
    if (not found_info or found_info[0] == 'N/A') and len(cs_str) >= 2:
        prefix = cs_str[:2]
        if prefix.isalnum():
            info = lookup_airline_info_by_code_inv(prefix)
            if info and info[0] != 'N/A': found_info = info
    return found_info if found_info and found_info[0] != 'N/A' else default


def create_investigation_tool_empty_map_figure(msg="No data", map_style="carto-positron"):
    map_font_clr = CHART_FONT_COLOR if map_style != "open-street-map" else 'black'
    layout = go.Layout(
        mapbox=dict(style=map_style, center={"lat": JUB_SPECIFIC_CENTER_LAT_INV, "lon": JUB_SPECIFIC_CENTER_LON_INV},
                    zoom=5),
        margin=dict(r=5, t=5, l=5, b=5), paper_bgcolor=CHART_PAPER_BG, plot_bgcolor=CHART_PLOT_BG,
        font=dict(color=map_font_clr))
    layout.annotations = [
        go.layout.Annotation(text=msg, align='center', showarrow=False, xref='paper', yref='paper', x=0.5, y=0.5,
                             font=dict(size=16, color=map_font_clr))]
    return go.Figure(data=[], layout=layout).to_dict()


def create_kpi_card_inv(title, val_id, tooltip):
    return dbc.Card([dbc.CardHeader(html.H6(title, className="text-light opacity-75 mb-0"), id=f"header-{val_id}"),
                     dbc.CardBody(html.H4(id=val_id, className="text-info text-center my-auto", children="-")),
                     dbc.Tooltip(tooltip, target=f"header-{val_id}", placement='bottom')],
                    className="h-100 d-flex flex-column")


def process_jub_billing_data_inv(df: pd.DataFrame) -> pd.DataFrame:
    df_proc = df.copy();
    df_proc['LATITUDE'] = pd.to_numeric(df_proc['LATITUDE'], errors='coerce');
    df_proc['LONGITUDE'] = pd.to_numeric(df_proc['LONGITUDE'], errors='coerce')
    df_proc.dropna(subset=['LATITUDE', 'LONGITUDE'], inplace=True)

    cond_spec = ((df_proc['LATITUDE'] >= JUB_SPECIFIC_MIN_LAT_INV) &
                 (df_proc['LATITUDE'] <= JUB_SPECIFIC_MAX_LAT_INV) &
                 (df_proc['LONGITUDE'] >= JUB_SPECIFIC_MIN_LON_INV) &
                 (df_proc['LONGITUDE'] <= JUB_SPECIFIC_MAX_LON_INV))

    cond_ss = ((df_proc['LATITUDE'] >= JUB_SOUTH_SUDAN_MIN_LAT_INV) &
               (df_proc['LATITUDE'] <= JUB_SOUTH_SUDAN_MAX_LAT_INV) &
               (df_proc['LONGITUDE'] >= JUB_SOUTH_SUDAN_MIN_LON_INV) &
               (df_proc['LONGITUDE'] <= JUB_SOUTH_SUDAN_MAX_LON_INV))

    df_proc['Appearance'] = "Not Passed";
    df_proc.loc[cond_ss & ~cond_spec, 'Appearance'] = "Investigate";
    df_proc.loc[cond_spec, 'Appearance'] = "Surely Passed"

    return df_proc


def create_visualizer_instance_layout(id_suffix=""):
    card_header_title = f"Callsign Flight Path Visualizer{(' - ' + id_suffix.replace('_', ' ').title()) if id_suffix else ''}"
    return dbc.Card([dbc.CardHeader(html.H5(card_header_title, className="fw-bold mb-0")),
                     dbc.CardBody([dbc.Row([dbc.Col(dbc.Input(id=f'callsign-input-for-map{id_suffix}',
                                                              placeholder='Enter Full Callsign (e.g., ETH302)',
                                                              type='text', value="ETH302"), md=3,
                                                    className="mb-2 mb-md-0"),
                                            dbc.Col([dbc.Input(id=f'callsign-num-pages-input{id_suffix}', type='number',
                                                               placeholder='Max Data Pages', min=1, value=1, step=1,
                                                               className="mb-0"),
                                                     dbc.Tooltip(
                                                         f"Number of data pages to fetch. Each page has up to {CALLSIGN_BATCH_SIZE} records.",
                                                         target=f'callsign-num-pages-input{id_suffix}',
                                                         placement='top')], md=2, className="mb-2 mb-md-0"),
                                            dbc.Col(dbc.Button('Load Flight Data',
                                                               id=f'load-callsign-data-button{id_suffix}', n_clicks=0,
                                                               color="primary"), md=3,
                                                    className="mb-2 mb-md-0"),
                                            dbc.Col(html.Div(id=f'callsign-map-status-message{id_suffix}',
                                                             className="mt-1 small"), md=4)],
                                           className="align-items-center mb-3 g-2"),
                                   html.Div([html.P("Filter Data by Date Range (available after loading data):",
                                                    className="mt-3 small text-muted"),
                                             dcc.RangeSlider(id=f'date-filter-slider{id_suffix}', min=0, max=0, step=1,
                                                             value=[0, 0], marks={}, allowCross=False,
                                                             tooltip={"placement": "bottom", "always_visible": False}),
                                             html.Div(id=f'date-filter-slider-output{id_suffix}',
                                                      className="small text-muted mt-1 text-center")],
                                            id=f'date-filter-slider-container{id_suffix}',
                                            style={'display': 'none', 'padding': '15px 25px'}),
                                   dcc.Loading(id=f"loading-callsign-map-and-table{id_suffix}", type="default",
                                               children=[
                                                   dcc.Graph(id=f'callsign-flight-map{id_suffix}',
                                                             figure=create_empty_map_figure_dict_viz(
                                                                 message="Enter callsign, max pages, and click 'Load Flight Data'."),
                                                             style={'height': '60vh'},
                                                             config={'displaylogo': False, 'scrollZoom': True}),
                                                   html.H5("Flight Data Points", className="mt-4 mb-3 text-info"),
                                                   dash_table.DataTable(id=f'callsign-path-data-table{id_suffix}',
                                                                        columns=INITIAL_TABLE_COLUMNS_VIZ, data=[],
                                                                        page_size=10,
                                                                        style_table={"overflowX": "auto",
                                                                                     "marginTop": "10px",
                                                                                     "maxHeight": "400px",
                                                                                     "minHeight": "150px",
                                                                                     "overflowY": "auto"},
                                                                        style_header={'backgroundColor': '#303030',
                                                                                      'color': CHART_FONT_COLOR,
                                                                                      'fontWeight': 'bold',
                                                                                      'border': f'1px solid {CHART_FONT_COLOR}40',
                                                                                      'position': 'sticky', 'top': 0,
                                                                                      'zIndex': 1},
                                                                        style_cell={'textAlign': 'left',
                                                                                    'padding': '10px',
                                                                                    'backgroundColor': '#23272B',
                                                                                    'color': CHART_FONT_COLOR,
                                                                                    'border': f'1px solid {CHART_FONT_COLOR}20',
                                                                                    'minWidth': '90px',
                                                                                    'whiteSpace': 'normal',
                                                                                    'height': 'auto',
                                                                                    'fontFamily': 'Roboto, sans-serif'},
                                                                        style_data_conditional=[
                                                                            {'if': {'row_index': 'odd'},
                                                                             'backgroundColor': '#2C3034'}
                                                                        ],
                                                                        export_format="csv", export_columns="all")])
                                   ])
                     ], className="mb-3")


def create_investigation_tool_layout(id_suffix="_inv_tool"):
    return dbc.Container([

        dbc.Row([dbc.Col(html.P(id=f'last-update-timestamp{id_suffix}',
                                className="text-center text-muted mb-1 small"), width=12),
                 dbc.Col(dbc.Alert(id=f'investigation-tool-status-alert{id_suffix}',
                                   children="Investigation tool status.", color="info", is_open=False,
                                   duration=7000, dismissable=True, className="small"), width=12)
                 ]),
        dbc.Card([
                  dbc.CardBody([
                      dbc.Row([dbc.Col(dcc.Loading(
                          children=dcc.Graph(id=f'jub-billing-map{id_suffix}')),
                          md=9,
                          className="mb-3 mb-md-0"),
                          dbc.Col([dbc.Card(
                              [dbc.CardHeader("Map Controls & Legend", className="small fw-bold"),
                               dbc.CardBody([
                                   dbc.Label("Manual Point Entry:", className="mt-2 small fw-bold"),
                                   dbc.Input(id=f'jub-manual-lat-input{id_suffix}', type='number',
                                             placeholder='Latitude (e.g, 7.86)', step=0.01,
                                             className="mb-2 form-control-sm"),
                                   dbc.Input(id=f'jub-manual-lon-input{id_suffix}', type='number',
                                             placeholder='Longitude (e.g, 29.69)', step=0.01,
                                             className="mb-2 form-control-sm"),
                                   dbc.Button('Add Point to Map', id=f'jub-add-point-button{id_suffix}',
                                              n_clicks=0, color="info", size="sm", className="w-100 mb-3"),
                                   dbc.Button('Clear Other Data (Show Manual Only)',
                                              id=f'jub-clear-auto-data-button{id_suffix}', color="warning", size="sm",
                                              className="w-100 mb-2 mt-2"),
                                   dbc.Button('Show All Map Data', id=f'jub-show-all-data-button{id_suffix}',
                                              color="secondary", size="sm", className="w-100 mb-3"),
                                   html.Div(id=f'jub-manual-point-info{id_suffix}',
                                            className="small text-muted mb-3 border-top pt-2 mt-2"),
                                   html.H6("Map Legend", className="mt-3 small fw-bold text-info"),
                                   html.Ul([html.Li(
                                       [html.I(className="fas fa-circle me-2", style={'color': '#28a745'}),
                                        "Surely Passed"]),
                                       html.Li([html.I(className="fas fa-circle me-2",
                                                       style={'color': '#ffc107'}), "Investigate"]),
                                       html.Li([html.I(className="fas fa-circle me-2",
                                                       style={'color': '#dc3545'}), "Not Passed"]),
                                       html.Li([html.I(className="fas fa-circle me-2",
                                                       style={'color': '#6f42c1'}), "New Entry"])],
                                       style={'listStyleType': 'none', 'paddingLeft': 0,
                                              'fontSize': '0.85em'})])], className="h-100")], md=3)],
                          className="mb-3"),
                      dbc.Row([dbc.Col(
                          create_kpi_card_inv("JUB FIR Flights (Live)", f"jub-fir-kpi-total{id_suffix}",
                                              "Total flights for JUB FIR billing from live data."), width=12,
                          md=4, className="mb-3"),
                          dbc.Col(
                              create_kpi_card_inv("Flights in Hot Zone", f"jub-fir-kpi-hotzone{id_suffix}",
                                                  "Flights 'Surely Passed' (specific hot zone)."),
                              width=12, md=4, className="mb-3"),
                          dbc.Col(create_kpi_card_inv("Flights to Investigate",
                                                      f"jub-fir-kpi-investigate{id_suffix}",
                                                      "Flights in JUB FIR general airspace, outside hot zone."),
                                  width=12, md=4, className="mb-3")], className="mb-3 g-3",
                          justify="center"),
                      dbc.Row([dbc.Col(html.H5("Billing Adjudication Table", className="text-info fw-bold mt-3 mb-3"),
                                       width=12)]),
                      dash_table.DataTable(id=f"jub-billing-data-table{id_suffix}", row_selectable='multi',
                                           selected_rows=[],
                                           style_table={"overflowX": "auto", "marginTop": "10px"},
                                           style_filter={'backgroundColor': 'rgba(102,255,255,0.1)',
                                                         'border': f'1px solid {CHART_FONT_COLOR}40',
                                                         'color': CHART_FONT_COLOR},
                                           style_header={'backgroundColor': '#303030', 'color': CHART_FONT_COLOR,
                                                         'fontWeight': 'bold',
                                                         'border': f'1px solid {CHART_FONT_COLOR}40'},
                                           style_cell={'textAlign': 'left', 'padding': '10px',
                                                       'backgroundColor': '#23272B',
                                                       'color': CHART_FONT_COLOR,
                                                       'border': f'1px solid {CHART_FONT_COLOR}20',
                                                       'minWidth': '100px',
                                                       'maxWidth': '200px', 'whiteSpace': 'normal',
                                                       'height': 'auto',
                                                       'fontFamily': 'Roboto, sans-serif'},
                                           style_data_conditional=[
                                               {'if': {'row_index': 'odd'}, 'backgroundColor': '#2C3034'},
                                               {'if': {'state': 'selected'},
                                                'backgroundColor': 'rgba(0,116,217,0.3)',
                                                'border': '1px solid #0074D9'},
                                               {'if': {'filter_query': '{Appearance} = "Surely Passed"',
                                                       'column_id': 'Appearance'},
                                                'backgroundColor': 'rgba(40,167,69,0.4)', 'color': 'white'},
                                               {'if': {'filter_query': '{Appearance} = "Investigate"',
                                                       'column_id': 'Appearance'},
                                                'backgroundColor': 'rgba(255,193,7,0.4)', 'color': 'black'},
                                               {'if': {'filter_query': '{Appearance} = "Not Passed"',
                                                       'column_id': 'Appearance'},
                                                'backgroundColor': 'rgba(220,53,69,0.4)', 'color': 'white'}],
                                           page_size=10, sort_action='native', filter_action='native', data=[],
                                           columns=[]),
                      dbc.Button("Download Selected Billing Rows",
                                 id=f"download-jub-selected-button{id_suffix}", color="success",
                                 className="mt-3", n_clicks=0),
                      dbc.Row(
                          [dbc.Col(html.H5("Airline Lookup Tool", className="text-info fw-bold mt-4 mb-2"), width=12)],
                          className="mt-3", justify="center"),
                      dbc.Row([dbc.Col([dbc.InputGroup([dbc.Input(id=f'airline-lookup-input{id_suffix}',
                                                                  placeholder='Enter Callsign Prefix (e.g., UAE, BAW, ET) or Code',
                                                                  type='text',
                                                                  className="form-control-sm"),
                                                        dbc.Button('Lookup Airline',
                                                                   id=f'airline-lookup-button{id_suffix}',
                                                                   n_clicks=0, color="info", size="sm")]),
                                        html.Div(id=f'airline-lookup-result{id_suffix}', className="mt-2",
                                                 style={'minHeight': '40px'})
                                        ], width=12, md=8, lg=6)], justify="center", className="mb-3")])
                  ], className="mb-3")
        ,
        dcc.Interval(id=f"interval-component{id_suffix}", interval=FETCH_INTERVAL_SECONDS_INV * 1000,
                     n_intervals=0)
    ], fluid=True, className="p-3")


app.layout = dbc.Container([
    dcc.Store(id='callsign-path-full-data-store'), dcc.Store(id='display-data-store'),
    dcc.Store(id='unique-dates-store'),
    dcc.Store(id='realtime-flight-data-store_inv_tool'), dcc.Store(id='processed-billing-data-store_inv_tool'),
    dcc.Store(id='jub-manual-points-store_inv_tool', data=[]),
    dcc.Store(id='manual-map-mode-store_inv_tool', data=False),
    dcc.Download(id="download-jub-selected-csv_inv_tool"),
    dcc.Download(id="download-html-report"),

    dbc.Row(
        [
            dbc.Col(html.Img(src=app.get_asset_url(LOGO_FILENAME),
                             style={'height': '45px'}) if APP_LOGO_B64 else html.Div(), width="auto"),
            dbc.Col(html.H1("Flight Operations Dashboard", className="text-primary my-0"), width=True),
            dbc.Col(dbc.Button("Download Full Report (HTML)", id="btn-download-report", color="info",
                               className="float-end"), width="auto", align="center")
        ],
        align="center",
        className="py-3 mb-4 mx-0 px-0",
        style={'borderBottom': f'2px solid {CHART_FONT_COLOR}50'}
    ),

    html.Div(dbc.Tabs(id="tabs-tools", active_tab="tab-tool1", children=[
        dbc.Tab(label="Callsign Path Visualizer", tab_id="tab-tool1",
                children=[html.Div(create_visualizer_instance_layout(id_suffix=""), className="p-md-3 p-1")]),
        dbc.Tab(label="Flight Investigation & Billing", tab_id="tab-tool2",
                children=[create_investigation_tool_layout(id_suffix="_inv_tool")])]), className="mb-4"),
    dbc.Row([dbc.Col(html.Footer(html.P(
        f"© {datetime.now(DISPLAY_TIMEZONE).year} Flight Operations Dashboard. Map image export requires 'kaleido' package. Airline data from OpenFlights.",
        className="text-center text-muted mt-4 small"), className="py-4"))])
], fluid=True, className="dbc")


def register_visualizer_callbacks(app_instance, id_suffix=""):
    @app_instance.callback(
        [Output(f'callsign-path-full-data-store{id_suffix}', 'data'),
         Output(f'callsign-map-status-message{id_suffix}', 'children'),
         Output(f'callsign-flight-map{id_suffix}', 'figure'),
         Output(f'callsign-path-data-table{id_suffix}', 'data'),
         Output(f'callsign-path-data-table{id_suffix}', 'columns'),
         Output(f'date-filter-slider-container{id_suffix}', 'style'),
         Output(f'load-callsign-data-button{id_suffix}', 'disabled')
         ],
        Input(f'load-callsign-data-button{id_suffix}', 'n_clicks'),
        State(f'callsign-input-for-map{id_suffix}', 'value'),
        State(f'callsign-num-pages-input{id_suffix}', 'value')
    )
    def load_callsign_data(n_clicks, callsign_input, num_pages_input):
        if not n_clicks or not callsign_input:
            if ctx.triggered_id is None and n_clicks is None:
                raise dash.exceptions.PreventUpdate("Initial call, no button click.")
            return (no_update, dbc.Alert("Callsign required.", color="warning", duration=4000, dismissable=True),
                    create_empty_map_figure_dict_viz("Enter callsign and pages, then load data."), [],
                    INITIAL_TABLE_COLUMNS_VIZ,
                    {'display': 'none'}, False)

        callsign_to_load = str(callsign_input).strip().upper()
        num_pages_to_fetch = 1
        if num_pages_input is not None:
            try:
                num_pages_to_fetch = int(num_pages_input)
                if num_pages_to_fetch < 1: num_pages_to_fetch = 1
            except ValueError:
                return (no_update, dbc.Alert("Invalid number of pages. Using 1.", color="warning", duration=4000,
                                             dismissable=True),
                        create_empty_map_figure_dict_viz("Invalid input."), [], INITIAL_TABLE_COLUMNS_VIZ,
                        {'display': 'none'}, False)

        status_msg = dbc.Alert(f"Loading data for {callsign_to_load} ({num_pages_to_fetch} pages)...", color="info")
        all_records = []
        error_occurred = False
        error_detail = ""

        for page_num in range(1, num_pages_to_fetch + 1):
            try:
                page_data, error_msg_detail = fetch_one_page_by_callsign_viz(
                    callsign_to_load, page_num, CALLSIGN_BATCH_SIZE, CALLSIGN_VISUALIZER_API_ENDPOINT
                )
                if error_msg_detail:
                    error_occurred = True;
                    error_detail = error_msg_detail
                    status_msg = dbc.Alert(f"Error fetching page {page_num} for {callsign_to_load}: {error_detail}",
                                           color="danger", duration=8000)
                    break
                if not page_data:
                    status_msg = dbc.Alert(
                        f"No more data from API after page {page_num - 1} for {callsign_to_load}. Loaded {len(all_records)} records.",
                        color="warning", duration=5000)
                    break
                all_records.extend(page_data)
            except Exception as e:
                error_occurred = True;
                error_detail = str(e)
                status_msg = dbc.Alert(f"Critical error during fetch for {callsign_to_load}: {error_detail}",
                                       color="danger", duration=8000)
                traceback.print_exc();
                break

        df_normalized = pd.DataFrame()
        if error_occurred or not all_records:
            if not all_records and not error_occurred: status_msg = dbc.Alert(
                f"No flight data found for {callsign_to_load}.", color="warning", duration=5000)
        else:
            df_normalized = normalize_flight_data_df_viz(pd.DataFrame(all_records), callsign_to_load)
            status_msg = dbc.Alert(
                f"Successfully loaded {len(df_normalized)} distinct records for {callsign_to_load} from {len(all_records)} raw points.",
                color="success", duration=5000)

        fig_dict, table_data, table_cols, final_data_dict = update_map_and_table_for_display_viz(
            df_normalized, callsign_to_load, "Data Loaded"
        )
        date_slider_style = {'display': 'block', 'padding': '15px 25px'} if not df_normalized.empty else {
            'display': 'none'}

        return (final_data_dict if not df_normalized.empty else None,
                status_msg, fig_dict, table_data, table_cols,
                date_slider_style, False)

    @app_instance.callback(
        Output(f'date-filter-slider{id_suffix}', 'min'), Output(f'date-filter-slider{id_suffix}', 'max'),
        Output(f'date-filter-slider{id_suffix}', 'value'), Output(f'date-filter-slider{id_suffix}', 'marks'),
        Output(f'date-filter-slider-container{id_suffix}', 'style', allow_duplicate=True),
        Output(f'unique-dates-store{id_suffix}', 'data'),
        Output(f'display-data-store{id_suffix}', 'data', allow_duplicate=True),
        Input(f'callsign-path-full-data-store{id_suffix}', 'data')
    )
    def setup_date_filter_slider(full_data_json):
        hide = {'display': 'none'}
        if not full_data_json: return 0, 0, [0, 0], {}, hide, None, None
        df = pd.DataFrame(full_data_json)
        if df.empty or 'DATE_ONLY' not in df.columns: return 0, 0, [0, 0], {}, hide, None, None
        df['DATE_ONLY'] = pd.to_datetime(df['DATE_ONLY'], errors='coerce').dt.date
        unique_dates = sorted(df['DATE_ONLY'].dropna().unique())
        if not unique_dates: return 0, 0, [0, 0], {}, hide, None, None
        min_idx, max_idx = 0, len(unique_dates) - 1;
        marks = {}
        if len(unique_dates) == 1:
            marks = {0: unique_dates[0].strftime('%Y-%m-%d')}
        elif len(unique_dates) <= 35:
            for i, d_obj in enumerate(unique_dates): marks[i] = d_obj.strftime('%d %b')
        else:
            num_m = min(10, len(unique_dates));
            step = max(1, math.ceil(len(unique_dates) / (num_m - 1 if num_m > 1 else 1)))
            m_idx_add = set(range(0, len(unique_dates), step));
            m_idx_add.update([0, len(unique_dates) - 1])
            for i_v in sorted(list(m_idx_add)):
                if int(i_v) < len(unique_dates): marks[int(i_v)] = unique_dates[int(i_v)].strftime('%Y-%m-%d')
        init_val = [min_idx, max_idx]
        return (min_idx, max_idx, init_val, marks, {'display': 'block', 'padding': '15px 25px'},
                [d.isoformat() for d in unique_dates], df.to_dict('records'))

    @app_instance.callback(
        Output(f'display-data-store{id_suffix}', 'data', allow_duplicate=True),
        Output(f'date-filter-slider-output{id_suffix}', 'children'),
        Input(f'date-filter-slider{id_suffix}', 'value'), State(f'unique-dates-store{id_suffix}', 'data'),
        State(f'callsign-path-full-data-store{id_suffix}', 'data')
    )
    def filter_data_by_date_range(slider_val, unique_dates_iso, full_data_json):
        if not slider_val or not unique_dates_iso or not full_data_json: return no_update, "Data missing for date filter."
        try:
            unique_dates = [date.fromisoformat(d) for d in unique_dates_iso]
        except Exception as e:
            return no_update, f"Date parsing error: {e}"

        if not (isinstance(slider_val, list) and len(slider_val) == 2 and
                all(isinstance(val, int) for val in slider_val)):
            return no_update, "Invalid slider values."

        idx_start, idx_end = slider_val[0], slider_val[1]
        if not (0 <= idx_start < len(unique_dates) and 0 <= idx_end < len(
                unique_dates) and idx_start <= idx_end): return no_update, "Invalid date indices from slider."

        start_date, end_date = unique_dates[idx_start], unique_dates[idx_end]
        df_full = pd.DataFrame(full_data_json)
        if df_full.empty: return [], "No data to filter."

        if 'DATE_ONLY' not in df_full.columns:
            if 'LAST_UPDATE_TIME' in df_full.columns:
                df_full['DATE_ONLY'] = pd.to_datetime(df_full['LAST_UPDATE_TIME'], errors='coerce', utc=True).dt.date
            else:
                return [], "Date information missing in full data."
        else:
            df_full['DATE_ONLY'] = pd.to_datetime(df_full['DATE_ONLY'], errors='coerce').dt.date

        df_filtered = df_full[(df_full['DATE_ONLY'] >= start_date) & (df_full['DATE_ONLY'] <= end_date)]
        date_range_txt = f"Displaying: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({len(df_filtered)} points)"
        return df_filtered.to_dict('records'), date_range_txt

    @app_instance.callback(
        [Output(f'callsign-flight-map{id_suffix}', 'figure', allow_duplicate=True),
         Output(f'callsign-path-data-table{id_suffix}', 'data', allow_duplicate=True),
         Output(f'callsign-path-data-table{id_suffix}', 'columns', allow_duplicate=True)],
        Input(f'display-data-store{id_suffix}', 'data'),
        State(f'callsign-input-for-map{id_suffix}', 'value')
    )
    def update_displays_from_filtered_data(display_json, callsign):
        callsign_title = str(callsign).strip().upper() if callsign else "N/A"
        if display_json is None:
            empty_fig = create_empty_map_figure_dict_viz(f"No data for {callsign_title}.")
            return empty_fig, [], INITIAL_TABLE_COLUMNS_VIZ

        df = pd.DataFrame(display_json)
        if df.empty:
            empty_fig = create_empty_map_figure_dict_viz(f"No data for {callsign_title} in selected range.")
            return empty_fig, [], INITIAL_TABLE_COLUMNS_VIZ

        if 'LAST_UPDATE_TIME' in df.columns:
            df['LAST_UPDATE_TIME'] = pd.to_datetime(df['LAST_UPDATE_TIME'], errors='coerce', utc=True)
            df.sort_values(by='LAST_UPDATE_TIME', ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)

        fig_dict, tbl_data, tbl_cols, _ = update_map_and_table_for_display_viz(df, callsign_title, "Filtered View")
        return fig_dict, tbl_data, tbl_cols


register_visualizer_callbacks(app, id_suffix="")

INV_SUFFIX = "_inv_tool"

@app.callback(
    [Output(f'realtime-flight-data-store{INV_SUFFIX}', 'data'),
     Output(f'last-update-timestamp{INV_SUFFIX}', 'children'),
     Output(f'investigation-tool-status-alert{INV_SUFFIX}', 'children'),
     Output(f'investigation-tool-status-alert{INV_SUFFIX}', 'color'),
     Output(f'investigation-tool-status-alert{INV_SUFFIX}', 'is_open')],
    Input(f"interval-component{INV_SUFFIX}", "n_intervals"),
    prevent_initial_call=False  # This callback should run on load
)
def update_realtime_data_store_inv(n_intervals):
    global fetcher_inv
    fetch_time = datetime.now(DISPLAY_TIMEZONE)
    status_msg_base = f"Last Live Check: {fetch_time.strftime('%Y-%m-%d %H:%M:%S %Z')}"
    alert_children = None;
    alert_color = "light";
    alert_is_open = False

    if fetcher_inv is None:
        alert_children = "Flight Data Fetcher not initialized. Live data unavailable."
        alert_color = "danger";
        alert_is_open = True
        return no_update, status_msg_base, alert_children, alert_color, alert_is_open
    try:
        df_batch = fetcher_inv.fetch_next_batch()
        if df_batch is None:
            alert_children = "Failed to fetch live data from API. Check API status or network.";
            alert_color = "warning";
            alert_is_open = True
            return no_update, status_msg_base, alert_children, alert_color, alert_is_open

        if df_batch.empty:
            last_ts_str = "initial setup"
            if fetcher_inv.last_processed_timestamp:
                last_ts_local = pytz.utc.localize(fetcher_inv.last_processed_timestamp).astimezone(DISPLAY_TIMEZONE)
                last_ts_str = last_ts_local.strftime('%Y-%m-%d %H:%M:%S %Z')
            status_msg_base = f"No New Live Data (since {last_ts_str}) | Last Check: {fetch_time.strftime('%H:%M:%S %Z')}"
            return no_update, status_msg_base, dash.no_update, dash.no_update, False

        df_proc = df_batch.copy();
        df_proc.columns = df_proc.columns.astype(str).str.lower().str.strip()
        col_map = {'lat': 'LATITUDE', 'lon': 'LONGITUDE', 'last_update': 'LAST_UPDATE_TIME',
                   'flight_id': FLIGHT_ID_COLUMN, 'model': 'AIRCRAFT_MODEL', 'alt': 'ALTITUDE', 'speed': 'SPEED',
                   'track': 'TRACK', 'callsign': 'FLIGHT_CALLSIGN', 'reg': 'REGISTRATION', 'origin': 'ORIGIN',
                   'destination': 'DESTINATION', 'flight': 'FLIGHT_NUMBER'}
        df_proc.rename(columns={k: v for k, v in col_map.items() if k in df_proc.columns}, inplace=True)

        ess_cols = ['LATITUDE', 'LONGITUDE', 'LAST_UPDATE_TIME', FLIGHT_ID_COLUMN]
        if 'FLIGHT_CALLSIGN' not in df_proc.columns: df_proc['FLIGHT_CALLSIGN'] = 'N/A'

        missing_ess = [c for c in ess_cols if c not in df_proc.columns]
        if missing_ess:
            print(f"Warning: INV LIVE batch missing essential columns: {missing_ess}. Skipping this batch.")
            alert_children = f"Live data batch has missing essential columns: {', '.join(missing_ess)}. Batch skipped."
            alert_color = "warning";
            alert_is_open = True;
            df_proc = pd.DataFrame()
        else:
            df_proc["LAST_UPDATE_TIME"] = pd.to_datetime(df_proc["LAST_UPDATE_TIME"], errors='coerce', utc=True)
            num_cols = ['LATITUDE', 'LONGITUDE', 'ALTITUDE', 'SPEED', 'TRACK'];
            str_cols = ['AIRCRAFT_MODEL', 'REGISTRATION', 'ORIGIN', 'DESTINATION', 'FLIGHT_NUMBER', 'FLIGHT_CALLSIGN']

            for col in num_cols:
                if col in df_proc.columns: df_proc[col] = pd.to_numeric(df_proc[col], errors='coerce')
            df_proc.dropna(subset=ess_cols, inplace=True)

            for col in str_cols:
                if col not in df_proc.columns: df_proc[col] = 'N/A'
                df_proc[col] = df_proc[col].astype(str).fillna('N/A').str.strip().replace(NA_REPLACE_VALUES_INV, 'N/A',
                                                                                          regex=False)
            if FLIGHT_ID_COLUMN in df_proc.columns:
                df_proc[FLIGHT_ID_COLUMN] = df_proc[FLIGHT_ID_COLUMN].astype(str).fillna('N/A').str.strip().replace(
                    NA_REPLACE_VALUES_INV, 'N/A', regex=False)
                df_proc = df_proc[df_proc[FLIGHT_ID_COLUMN] != 'N/A']
            else:
                df_proc = pd.DataFrame()

        if df_proc.empty and not alert_is_open:
            alert_children = "No valid live data after processing current batch.";
            alert_color = "info";
            alert_is_open = True
            return no_update, status_msg_base, alert_children, alert_color, alert_is_open

        if 'LAST_UPDATE_TIME' in df_proc.columns and pd.api.types.is_datetime64_any_dtype(df_proc['LAST_UPDATE_TIME']):
            if df_proc['LAST_UPDATE_TIME'].dt.tz is None:
                df_proc['LAST_UPDATE_TIME'] = df_proc['LAST_UPDATE_TIME'].dt.tz_localize('UTC')
            else:
                df_proc['LAST_UPDATE_TIME'] = df_proc['LAST_UPDATE_TIME'].dt.tz_convert('UTC')
            df_proc['LAST_UPDATE_TIME'] = df_proc['LAST_UPDATE_TIME'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        else:
            df_proc['LAST_UPDATE_TIME'] = None

        data_store = df_proc.replace({pd.NA: None, pd.NaT: None, np.nan: None}).to_dict('records')

        final_status_msg = f"Live View Updated ({fetch_time.strftime('%H:%M:%S')}): {len(data_store)} new data points processed."
        if alert_is_open:
            alert_children = html.Div([html.P(alert_children), html.Hr(), html.P(final_status_msg)])
        else:
            alert_children = final_status_msg;
            alert_color = "success";
            alert_is_open = True

        return data_store, status_msg_base, alert_children, alert_color, alert_is_open

    except Exception as e:
        print(f"CRITICAL ERROR in update_realtime_data_store_inv: {e}");
        traceback.print_exc()
        alert_children = f"An unexpected error occurred while fetching/processing live data: {str(e)}"
        alert_color = "danger";
        alert_is_open = True
        return no_update, status_msg_base, alert_children, alert_color, alert_is_open


@app.callback(
    [Output(f'processed-billing-data-store{INV_SUFFIX}', 'data'),
     Output(f'jub-fir-kpi-total{INV_SUFFIX}', 'children'), Output(f'jub-fir-kpi-hotzone{INV_SUFFIX}', 'children'),
     Output(f'jub-fir-kpi-investigate{INV_SUFFIX}', 'children'),
     Output(f'investigation-tool-status-alert{INV_SUFFIX}', 'children', allow_duplicate=True),
     Output(f'investigation-tool-status-alert{INV_SUFFIX}', 'color', allow_duplicate=True),
     Output(f'investigation-tool-status-alert{INV_SUFFIX}', 'is_open', allow_duplicate=True)],
    Input(f"realtime-flight-data-store{INV_SUFFIX}", "data")
)
def update_processed_billing_data_and_kpis_inv(stored_data):
    if not stored_data: return no_update, "-", "-", "-", dash.no_update, dash.no_update, dash.no_update
    try:
        df_rt = pd.DataFrame(stored_data);
        if df_rt.empty: return no_update, "-", "-", "-", "No realtime data to process for billing.", "info", True

        req_cols = [FLIGHT_ID_COLUMN, 'LAST_UPDATE_TIME', 'LATITUDE', 'LONGITUDE',
                    'AIRCRAFT_MODEL', 'ORIGIN', 'DESTINATION', 'FLIGHT_CALLSIGN',
                    'ALTITUDE', 'REGISTRATION']
        for col in req_cols:
            if col not in df_rt.columns:
                if col in ['LATITUDE', 'LONGITUDE', 'ALTITUDE']:
                    df_rt[col] = np.nan
                elif col == 'LAST_UPDATE_TIME':
                    df_rt[col] = pd.NaT
                else:
                    df_rt[col] = 'N/A'

        df_rt['LAST_UPDATE_TIME'] = pd.to_datetime(df_rt['LAST_UPDATE_TIME'], errors='coerce', utc=True)
        df_rt.dropna(subset=[FLIGHT_ID_COLUMN, 'LAST_UPDATE_TIME', 'LATITUDE', 'LONGITUDE'], inplace=True)
        df_rt[FLIGHT_ID_COLUMN] = df_rt[FLIGHT_ID_COLUMN].astype(str).fillna('N/A');
        df_rt = df_rt[df_rt[FLIGHT_ID_COLUMN] != 'N/A']

        if df_rt.empty: return no_update, "-", "-", "-", "No valid realtime data after initial cleaning for billing.", "info", True

        df_latest_daily = pd.DataFrame()
        try:
            if not pd.api.types.is_datetime64_any_dtype(df_rt['LAST_UPDATE_TIME']):
                df_rt['LAST_UPDATE_TIME'] = pd.to_datetime(df_rt['LAST_UPDATE_TIME'], errors='coerce', utc=True)
            df_rt.dropna(subset=['LAST_UPDATE_TIME'], inplace=True)

            if not df_rt.empty:
                df_rt['DATE'] = df_rt['LAST_UPDATE_TIME'].dt.tz_convert(DISPLAY_TIMEZONE).dt.date
                latest_idx = df_rt.loc[df_rt.groupby([FLIGHT_ID_COLUMN, 'DATE'])["LAST_UPDATE_TIME"].idxmax()].index;
                df_latest_daily = df_rt.loc[latest_idx].copy()
            else:
                df_latest_daily = pd.DataFrame()
        except Exception as e:
            print(
                f"Warning: INV JUB Billing daily latest logic failed ({e}), using overall latest if any data remains.");
            if not df_rt.empty:
                latest_idx = df_rt.loc[df_rt.groupby(FLIGHT_ID_COLUMN)["LAST_UPDATE_TIME"].idxmax()].index;
                df_latest_daily = df_rt.loc[latest_idx].copy()
            else:
                df_latest_daily = pd.DataFrame()

        if df_latest_daily.empty: return no_update, "-", "-", "-", "No data after daily aggregation for billing.", "info", True

        df_appearance = process_jub_billing_data_inv(df_latest_daily)

        if 'FLIGHT_CALLSIGN' in df_appearance.columns:
            airline_info = df_appearance['FLIGHT_CALLSIGN'].apply(get_airline_info_from_callsign_inv)
            df_appearance['AIRLINE_NAME'] = [info[0] for info in airline_info];
            df_appearance['AIRLINE_COUNTRY'] = [info[1] for info in airline_info]
        else:
            df_appearance['AIRLINE_NAME'] = 'N/A';
            df_appearance['AIRLINE_COUNTRY'] = 'N/A'

        total = len(df_appearance)
        passed = len(df_appearance[df_appearance['Appearance'] == "Surely Passed"])
        investigate = len(df_appearance[df_appearance['Appearance'] == "Investigate"])

        disp_cols_check = [FLIGHT_ID_COLUMN, 'FLIGHT_CALLSIGN', 'AIRCRAFT_MODEL', 'REGISTRATION', 'ORIGIN',
                           'DESTINATION', 'LATITUDE', 'LONGITUDE', 'ALTITUDE', 'LAST_UPDATE_TIME', 'Appearance',
                           'AIRLINE_NAME', 'AIRLINE_COUNTRY']
        for col in disp_cols_check:
            if col not in df_appearance.columns:
                if col in ['LATITUDE', 'LONGITUDE', 'ALTITUDE']:
                    df_appearance[col] = np.nan
                elif col == 'LAST_UPDATE_TIME':
                    df_appearance[col] = pd.NaT if 'LAST_UPDATE_TIME' not in df_appearance or pd.isnull(
                        df_appearance['LAST_UPDATE_TIME']).all() else df_appearance['LAST_UPDATE_TIME']
                else:
                    df_appearance[col] = 'N/A'

        if 'LAST_UPDATE_TIME' in df_appearance.columns and pd.api.types.is_datetime64_any_dtype(
                df_appearance['LAST_UPDATE_TIME']):
            df_appearance['LAST_UPDATE_TIME'] = df_appearance['LAST_UPDATE_TIME'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')

        processed_recs = df_appearance.replace({pd.NA: None, np.nan: None, pd.NaT: None}).to_dict('records')

        return (processed_recs, f"{total:,}", f"{passed:,}", f"{investigate:,}", "Billing data processed successfully.",
                "success", True)

    except Exception as e:
        print(
            f"Error update_processed_billing_data_and_kpis_inv: {e}");
        traceback.print_exc();
        return no_update, "-", "-", "-", f"Error processing billing data: {str(e)}", "danger", True


@app.callback(
    Output(f'jub-billing-data-table{INV_SUFFIX}', 'data'), Output(f'jub-billing-data-table{INV_SUFFIX}', 'columns'),
    Input(f'processed-billing-data-store{INV_SUFFIX}', 'data')
)
def update_jub_billing_table_inv(processed_data):
    if not processed_data: return [], []
    try:
        df_billing = pd.DataFrame(processed_data);
        if df_billing.empty: return [], []

        jub_cols_ordered = [FLIGHT_ID_COLUMN, 'FLIGHT_CALLSIGN', 'AIRLINE_NAME', 'AIRLINE_COUNTRY', 'AIRCRAFT_MODEL',
                            'REGISTRATION', 'ORIGIN', 'DESTINATION', 'LATITUDE', 'LONGITUDE', 'ALTITUDE',
                            'LAST_UPDATE_TIME', 'Appearance']

        for col in jub_cols_ordered:
            if col not in df_billing.columns:
                if col in ['LATITUDE', 'LONGITUDE', 'ALTITUDE']:
                    df_billing[col] = np.nan
                elif col == 'LAST_UPDATE_TIME':
                    df_billing[col] = None
                else:
                    df_billing[col] = 'N/A'

        df_display = df_billing[jub_cols_ordered].copy()

        if 'LAST_UPDATE_TIME' in df_display.columns:
            df_display['LAST_UPDATE_TIME_obj'] = pd.to_datetime(df_display['LAST_UPDATE_TIME'], errors='coerce',
                                                                utc=True)
            df_display['LAST_UPDATE_TIME_DISPLAY'] = df_display['LAST_UPDATE_TIME_obj'].apply(
                lambda x: x.tz_convert(DISPLAY_TIMEZONE).strftime('%Y-%m-%d %H:%M:%S') if pd.notnull(x) else 'N/A'
            )
        else:
            df_display['LAST_UPDATE_TIME_DISPLAY'] = 'N/A'

        jub_columns = []
        for col_id in jub_cols_ordered:
            name = col_id.replace('_', ' ').title()
            if col_id == FLIGHT_ID_COLUMN: name = "Flight ID"
            if col_id == 'AIRLINE_NAME': name = "Airline"
            if col_id == 'AIRLINE_COUNTRY': name = "Country"

            current_col_id_for_data = col_id
            if col_id == 'LAST_UPDATE_TIME':
                name = 'Last Seen (EAT)';
                current_col_id_for_data = 'LAST_UPDATE_TIME_DISPLAY'

            jub_columns.append({"name": name, "id": current_col_id_for_data, "editable": False})

        cols_for_table_data = [c['id'] for c in jub_columns]
        temp_df_for_dict = pd.DataFrame()
        for c_id in cols_for_table_data:
            if c_id in df_display.columns:
                temp_df_for_dict[c_id] = df_display[c_id]
            else:
                temp_df_for_dict[c_id] = 'N/A'

        df_dict = temp_df_for_dict.replace({pd.NA: None, np.nan: None, pd.NaT: None}).to_dict("records")

        return df_dict, jub_columns
    except Exception as e:
        print(f"Error update_jub_billing_table_inv: {e}");
        traceback.print_exc();
        return [], []


@app.callback(
    Output(f'jub-billing-map{INV_SUFFIX}', 'figure'),
    Output(f'jub-manual-points-store{INV_SUFFIX}', 'data'),
    Output(f'jub-manual-point-info{INV_SUFFIX}', 'children'),
    Output(f'jub-manual-lat-input{INV_SUFFIX}', 'value'),
    Output(f'jub-manual-lon-input{INV_SUFFIX}', 'value'),
    Output(f'manual-map-mode-store{INV_SUFFIX}', 'data'),
    # Removed allow_duplicate=True, ensure this is the only updater
    Input(f'processed-billing-data-store{INV_SUFFIX}', 'data'),
    Input(f'jub-add-point-button{INV_SUFFIX}', 'n_clicks'),
    Input(f"jub-billing-data-table{INV_SUFFIX}", "selected_rows"),
    Input(f'jub-clear-auto-data-button{INV_SUFFIX}', 'n_clicks'),
    Input(f'jub-show-all-data-button{INV_SUFFIX}', 'n_clicks'),
    State(f'jub-manual-lat-input{INV_SUFFIX}', 'value'),
    State(f'jub-manual-lon-input{INV_SUFFIX}', 'value'),
    State(f'jub-manual-points-store{INV_SUFFIX}', 'data'),
    State(f"jub-billing-data-table{INV_SUFFIX}", "data"),
    State(f'manual-map-mode-store{INV_SUFFIX}', 'data')
)
def update_jub_billing_map_inv(processed_data, n_clicks_add, sel_rows,
                               n_clicks_clear_auto, n_clicks_show_all,
                               man_lat, man_lon, exist_man_pts, tbl_data_state,
                               current_manual_mode):
    triggered_id = ctx.triggered_id if ctx.triggered_id else 'NoTrigger'
    new_pt_info_children = []
    clear_lat_input = dash.no_update
    clear_lon_input = dash.no_update

    map_font_clr = CHART_FONT_COLOR if JUB_BILLING_MAP_STYLE_INV != "open-street-map" else 'black'

    updated_manual_pts = exist_man_pts if exist_man_pts is not None else []
    new_manual_mode_status = bool(current_manual_mode) if current_manual_mode is not None else False

    if triggered_id == f'jub-add-point-button{INV_SUFFIX}' and n_clicks_add > 0:
        if man_lat is not None and man_lon is not None:
            try:
                lat, lon = float(man_lat), float(man_lon)
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    new_pt_id = f'manual_{lat:.4f}_{lon:.4f}_{n_clicks_add}_{datetime.now().timestamp()}'
                    new_manual_point = {
                        FLIGHT_ID_COLUMN: new_pt_id, 'LATITUDE': lat, 'LONGITUDE': lon,
                        'Appearance': 'new entry', 'FLIGHT_CALLSIGN': 'Manual Entry',
                        'ALTITUDE': 15000, 'AIRLINE_NAME': 'N/A', 'AIRLINE_COUNTRY': 'N/A',
                        'REGISTRATION': 'N/A', 'AIRCRAFT_MODEL': 'N/A',
                        'ORIGIN': 'N/A', 'DESTINATION': 'N/A',
                        'LAST_UPDATE_TIME': datetime.now(timezone.utc).isoformat()
                    }
                    updated_manual_pts.append(new_manual_point)
                    clear_lat_input, clear_lon_input = '', ''

                    new_pt_info_children.append(html.H6("Last Manually Added Point:", className="text-info small"))
                    new_pt_info_children.append(html.P(f"Lat: {lat:.4f}, Lon: {lon:.4f}", className="small mb-0"))
                    new_pt_info_children.append(html.P(f"Status: New Entry", className="small mb-0"))

                else:
                    new_pt_info_children = dbc.Alert("Error: Latitude/Longitude values are out of valid range.",
                                                     color="danger", dismissable=True, duration=4000)
            except ValueError:
                new_pt_info_children = dbc.Alert("Error: Invalid latitude/longitude format. Please enter numbers.",
                                                 color="danger", dismissable=True, duration=4000)
        else:
            new_pt_info_children = dbc.Alert("Please enter both latitude and longitude.", color="warning",
                                             dismissable=True, duration=4000)

    if triggered_id == f'jub-clear-auto-data-button{INV_SUFFIX}':
        new_manual_mode_status = True
    elif triggered_id == f'jub-show-all-data-button{INV_SUFFIX}':
        new_manual_mode_status = False
    elif triggered_id == f'jub-add-point-button{INV_SUFFIX}' and n_clicks_add > 0:
        new_manual_mode_status = True

    df_manual_pts_to_plot = pd.DataFrame(updated_manual_pts) if updated_manual_pts else pd.DataFrame()
    plot_df = pd.DataFrame()

    if new_manual_mode_status:
        plot_df = df_manual_pts_to_plot
    else:
        df_base = pd.DataFrame(processed_data) if processed_data else pd.DataFrame()
        df_selected_from_table = pd.DataFrame()

        if sel_rows and tbl_data_state:
            df_current_table_view = pd.DataFrame(tbl_data_state)
            if not df_current_table_view.empty and FLIGHT_ID_COLUMN in df_current_table_view.columns:
                selected_flight_ids = {
                    df_current_table_view.iloc[i][FLIGHT_ID_COLUMN] for i in sel_rows
                    if i < len(df_current_table_view) and pd.notna(
                        df_current_table_view.iloc[i].get(FLIGHT_ID_COLUMN))
                }
                if selected_flight_ids and not df_base.empty:
                    df_selected_from_table = df_base[df_base[FLIGHT_ID_COLUMN].isin(selected_flight_ids)].copy()

        if not df_selected_from_table.empty:
            plot_df = df_selected_from_table
        elif not df_base.empty:
            plot_df = df_base

        if not df_manual_pts_to_plot.empty:
            if not plot_df.empty:
                plot_df = pd.concat([plot_df, df_manual_pts_to_plot]).drop_duplicates(subset=[FLIGHT_ID_COLUMN],
                                                                                      keep='first').reset_index(
                    drop=True)
            else:
                plot_df = df_manual_pts_to_plot

    if plot_df.empty:
        map_message = "No JUB FIR Data Matching Selection"
        if new_manual_mode_status and (not updated_manual_pts or len(updated_manual_pts) == 0):
            map_message = "No manual points entered. Add points or click 'Show All Map Data'."

        return (create_investigation_tool_empty_map_figure(map_message, map_style=JUB_BILLING_MAP_STYLE_INV),
                updated_manual_pts, new_pt_info_children or no_update, clear_lat_input, clear_lon_input,
                new_manual_mode_status)

    plot_cols_with_defaults = {
        'LATITUDE': np.nan, 'LONGITUDE': np.nan, 'Appearance': 'N/A',
        'ALTITUDE': 10000, 'FLIGHT_CALLSIGN': 'N/A', 'REGISTRATION': 'N/A',
        'AIRCRAFT_MODEL': 'N/A', 'ORIGIN': 'N/A', 'DESTINATION': 'N/A',
        FLIGHT_ID_COLUMN: lambda: f"unknown_{datetime.now().timestamp()}",
        'LAST_UPDATE_TIME': pd.NaT, 'AIRLINE_NAME': 'N/A', 'AIRLINE_COUNTRY': 'N/A'
    }
    for col, default_val_or_func in plot_cols_with_defaults.items():
        if col not in plot_df.columns:
            plot_df[col] = default_val_or_func() if callable(default_val_or_func) else default_val_or_func
        elif col in ['ALTITUDE', 'Appearance', 'REGISTRATION', 'AIRCRAFT_MODEL', 'ORIGIN', 'DESTINATION',
                     'AIRLINE_NAME', 'AIRLINE_COUNTRY', 'FLIGHT_CALLSIGN']:
            plot_df[col] = plot_df[col].fillna('N/A')

    plot_df.dropna(subset=['LATITUDE', 'LONGITUDE'], inplace=True)
    if plot_df.empty:
        return (create_investigation_tool_empty_map_figure("No Valid Coordinates in Data for Map",
                                                           map_style=JUB_BILLING_MAP_STYLE_INV),
                updated_manual_pts, new_pt_info_children or no_update, clear_lat_input, clear_lon_input,
                new_manual_mode_status)

    plot_df['ALTITUDE_plot'] = pd.to_numeric(plot_df['ALTITUDE'], errors='coerce').fillna(10000)
    plot_df['FLIGHT_CALLSIGN_display'] = plot_df['FLIGHT_CALLSIGN'].fillna('N/A').astype(str)
    plot_df['AIRLINE_NAME_display'] = plot_df['AIRLINE_NAME'].fillna('N/A').astype(str)
    plot_df['AIRLINE_COUNTRY_display'] = plot_df['AIRLINE_COUNTRY'].fillna('N/A').astype(str)
    plot_df['LAST_UPDATE_TIME_display'] = pd.to_datetime(plot_df['LAST_UPDATE_TIME'], errors='coerce',
                                                         utc=True).apply(
        lambda x: x.tz_convert(DISPLAY_TIMEZONE).strftime('%Y-%m-%d %H:%M') if pd.notnull(x) else 'N/A'
    )

    custom_data_list_for_map = [
        plot_df['AIRLINE_NAME_display'], plot_df['AIRLINE_COUNTRY_display'],
        plot_df['ALTITUDE'], plot_df['Appearance'], plot_df['REGISTRATION'],
        plot_df['LAST_UPDATE_TIME_display'], plot_df['AIRCRAFT_MODEL'],
        plot_df['ORIGIN'], plot_df['DESTINATION']
    ]

    fig_billing = px.scatter_mapbox(plot_df,
                                    lat="LATITUDE", lon="LONGITUDE", color="Appearance",
                                    size="ALTITUDE_plot", size_max=18,
                                    hover_name="FLIGHT_CALLSIGN_display",
                                    custom_data=custom_data_list_for_map,
                                    color_discrete_map={"Surely Passed": "#28a745", "Investigate": "#ffc107",
                                                        "Not Passed": "#dc3545", "new entry": "#6f42c1",
                                                        "N/A": "#808080"},
                                    zoom=4.5,
                                    center={"lat": JUB_SPECIFIC_CENTER_LAT_INV, "lon": JUB_SPECIFIC_CENTER_LON_INV},
                                    height=600)
    fig_billing.update_traces(
        hovertemplate="<br>".join([
            "<b>%{hovertext}</b>",
            "Airline: %{customdata[0]} (%{customdata[1]})",
            "Lat: %{lat:.4f}, Lon: %{lon:.4f}",
            "Alt: %{customdata[2]:.0f} ft",
            "Status: %{customdata[3]}",
            "Reg: %{customdata[4]}",
            "Aircraft: %{customdata[6]}",
            "Origin: %{customdata[7]}",
            "Dest: %{customdata[8]}",
            "Last Seen: %{customdata[5]}",
            "<extra></extra>"
        ])
    )

    fig_billing.update_layout(
        mapbox_style=JUB_BILLING_MAP_STYLE_INV,
        margin={"r": 10, "t": 60, "l": 10, "b": 10},
        font=dict(color=map_font_clr),
        uirevision=f"{str(sel_rows)}-{len(updated_manual_pts)}-{JUB_BILLING_MAP_STYLE_INV}-{new_manual_mode_status}"
    )
    fig_billing.update_traces(marker=dict(sizemin=5))

    return fig_billing, updated_manual_pts, new_pt_info_children or no_update, clear_lat_input, clear_lon_input, new_manual_mode_status


@app.callback(
    Output(f"download-jub-selected-csv{INV_SUFFIX}", "data"),
    Input(f"download-jub-selected-button{INV_SUFFIX}", "n_clicks"),
    State(f"jub-billing-data-table{INV_SUFFIX}", "selected_rows"), State(f"jub-billing-data-table{INV_SUFFIX}", "data")
)
def download_selected_jub_billing_csv_inv(n_clicks, sel_indices, all_data):
    if n_clicks is None or n_clicks == 0 or not sel_indices or not all_data: return dash.no_update
    try:
        valid_sel_indices = [i for i in sel_indices if i < len(all_data)]
        if not valid_sel_indices: return dash.no_update

        sel_data = [all_data[i] for i in valid_sel_indices];
        if not sel_data: return dash.no_update

        df_sel = pd.DataFrame(sel_data);
        ts = datetime.now(DISPLAY_TIMEZONE).strftime("%Y%m%d_%H%M%S")
        return dcc.send_data_frame(df_sel.to_csv, filename=f"jub_billing_selected_{ts}.csv", index=False,
                                   encoding='utf-8-sig')
    except Exception as e:
        print(f"Error INV CSV download: {e}");
        traceback.print_exc();
        return dash.no_update


@app.callback(
    Output(f'airline-lookup-result{INV_SUFFIX}', 'children'), Input(f'airline-lookup-button{INV_SUFFIX}', 'n_clicks'),
    State(f'airline-lookup-input{INV_SUFFIX}', 'value')
)
def display_airline_lookup_result_inv(n_clicks, code_input):
    if not code_input: return dbc.Alert("Please enter a callsign prefix or IATA/ICAO code.", color="warning",
                                        dismissable=True, duration=5000)

    code_proc = str(code_input).strip().upper()
    if not code_proc: return dbc.Alert("Input cannot be empty.", color="warning", dismissable=True, duration=5000)

    if not iata_to_airline_map and not icao_to_airline_map: return dbc.Alert(
        f"Airline reference data is currently unavailable. Cannot lookup '{code_proc}'.", color="danger",
        dismissable=True, duration=8000)

    airline_info = lookup_airline_info_by_code_inv(code_proc)

    if not airline_info or airline_info[0] == 'N/A':
        temp_name, temp_country = 'N/A', 'N/A'
        if len(code_proc) >= 3:
            info3 = lookup_airline_info_by_code_inv(code_proc[:3])
            if info3 and info3[0] != 'N/A':
                temp_name, temp_country = info3
        if temp_name == 'N/A' and len(code_proc) >= 2:
            info2 = lookup_airline_info_by_code_inv(code_proc[:2])
            if info2 and info2[0] != 'N/A':
                temp_name, temp_country = info2
        airline_info = (temp_name, temp_country)

    if airline_info and airline_info[0] != 'N/A':
        name, country = airline_info
        country_disp = f", Country: {country}" if country and country != 'N/A' else ""
        return dbc.Alert(
            [html.Strong(f"Input: {code_input.upper()}"), html.Br(), f" Airline: {name}{country_disp}"],
            color="success", dismissable=True, duration=10000
        )
    else:
        return dbc.Alert(f"No airline found for '{code_input.upper()}'. Try a 2 or 3 letter prefix.", color="danger",
                         dismissable=True, duration=8000)


@app.callback(
    [Output("download-html-report", "data"),
     Output("btn-download-report", "children"),
     Output("btn-download-report", "disabled")],
    Input("btn-download-report", "n_clicks"),
    [State("callsign-flight-map", "figure"),
     State("callsign-path-data-table", "data"),
     State('date-filter-slider-output', 'children'),
     State(f"jub-billing-map{INV_SUFFIX}", "figure"),
     State(f"jub-billing-data-table{INV_SUFFIX}", "data"),
     State(f"jub-billing-data-table{INV_SUFFIX}", "selected_rows"),
     State(f'manual-map-mode-store{INV_SUFFIX}', 'data'),
     State(f'jub-manual-points-store{INV_SUFFIX}', 'data')
     ]
)
def download_html_report(n_clicks_report, fig_viz_data, table_viz_data,
                         date_filter_summary_viz,
                         fig_inv_data, table_inv_data_full, selected_inv_rows,
                         manual_map_mode, manual_points_data):
    if not n_clicks_report:  # Handles initial call if prevent_initial_callbacks=False for this callback
        if ctx.triggered_id is None:
            raise dash.exceptions.PreventUpdate("Initial call, no button click.")
        return dash.no_update, "Download Full Report (HTML)", False

    report_logo_b64 = APP_LOGO_B64

    def df_to_html_table(df_data, table_id="table", caption=None):
        if df_data is None:
            return f"<p style='text-align:center; font-style:italic; color:#555;'>{caption if caption else 'Table Data'}: No data available (Data is None).</p>"
        if not isinstance(df_data, pd.DataFrame):
            if isinstance(df_data, list):
                try:
                    df = pd.DataFrame(df_data)
                except Exception as e:
                    print(f"Error converting list to DataFrame in df_to_html_table for table '{table_id}': {e}")
                    return f"<p style='text-align:center; font-style:italic; color:red;'>{caption if caption else 'Table Data'}: Error processing list data.</p>"
            else:
                print(f"Warning: Unexpected data type for df_to_html_table (table_id: {table_id}): {type(df_data)}")
                return f"<p style='text-align:center; font-style:italic; color:red;'>{caption if caption else 'Table Data'}: Invalid data type received.</p>"
        else:
            df = df_data.copy()
        if df.empty:
            return f"<p style='text-align:center; font-style:italic; color:#555;'>{caption if caption else 'Table Data'}: No data available (Table is empty).</p>"

        df.columns = [
            str(col).replace('_DISPLAY', '').replace('_', ' ').title() if isinstance(col, str) else str(col).title() for
            col in df.columns]
        html_table = df.to_html(classes="table table-striped table-bordered table-sm table-hover", border=0,
                                index=False,
                                table_id=table_id, na_rep='N/A')
        if caption:
            html_table = f"<h4 style='text-align:left; margin-top:20px; margin-bottom:10px; color: #004085; font-weight:600;'>{caption}</h4>" + html_table
        return html_table

    def fig_to_base64_img_html(fig_data, width_percent=95, height_px=450, title="Map View"):
        no_data_message = f"<p style='text-align:center; font-style:italic; color:#555;'>{title}: Map data not available for report.</p>"
        if not fig_data or not fig_data.get('data'):
            if title == "Billing Adjudication Map" and manual_map_mode and (
                    not manual_points_data or len(manual_points_data) == 0):
                return f"<p style='text-align:center; font-style:italic; color:#555;'>{title}: Manual points mode active, but no manual points to display.</p>"
            return no_data_message
        try:
            fig = go.Figure(fig_data);
            fig.update_layout(
                paper_bgcolor='white',
                plot_bgcolor='white',
                font_color='black',
                title=dict(text=title, x=0.5, font=dict(color='black', size=16))
            )
            if fig.layout.mapbox:
                fig.layout.mapbox.style = "carto-positron"
                if fig.layout.mapbox.center:
                    fig.layout.mapbox.zoom = fig.layout.mapbox.zoom if fig.layout.mapbox.zoom else 5
                else:
                    fig.layout.mapbox.center = {"lat": SOUTH_SUDAN_CENTER_LAT_VIZ, "lon": SOUTH_SUDAN_CENTER_LON_VIZ}
                    fig.layout.mapbox.zoom = 4
            img_bytes = fig.to_image(format="png", width=1000, height=height_px, scale=2);
            img_base64 = base64.b64encode(img_bytes).decode()
            return f'<div style="text-align:center; padding:10px 0;"><img src="data:image/png;base64,{img_base64}" alt="{title}" class="map-image" style="width:{width_percent}%;height:auto;max-height:{height_px}px;border:1px solid #ccc;margin-top:10px;margin-bottom:10px;border-radius:4px;"></div>'
        except ImportError:
            print("Kaleido package not installed. Map images cannot be generated for the report.")
            return f"<p style='text-align:center; font-style:italic; color:red;'>Map image generation failed: Kaleido package not installed.</p>"
        except Exception as e:
            print(f"Error converting figure to image for report: {e}");
            traceback.print_exc()
            return f"<p style='text-align:center; font-style:italic; color:red;'>Error generating map image: {str(e)}</p>"

    billing_table_caption_text = "Billing Adjudication Data"
    df_billing_report_final_for_html = pd.DataFrame()

    if manual_map_mode and manual_points_data:
        df_billing_report_final_for_html = pd.DataFrame(manual_points_data)
        billing_table_caption_text = "Manual Points Data"
        if 'Appearance' in df_billing_report_final_for_html.columns:
            df_billing_report_final_for_html = df_billing_report_final_for_html.drop(columns=['Appearance'])
    elif table_inv_data_full:  # This is list of dicts from datatable
        df_inv_full_for_report = pd.DataFrame(table_inv_data_full)  # Convert to DF
        if not df_inv_full_for_report.empty:
            temp_billing_df = pd.DataFrame()  # Ensure it's a DF for processing
            if selected_inv_rows:
                valid_selected_indices = [i for i in selected_inv_rows if i < len(df_inv_full_for_report)]
                if valid_selected_indices:
                    temp_billing_df = df_inv_full_for_report.iloc[valid_selected_indices].copy()
                    billing_table_caption_text = "Billing Adjudication Data (Selected Rows)"
                else:  # No valid selection, use all from table_inv_data_full
                    temp_billing_df = df_inv_full_for_report.copy()
                    billing_table_caption_text = "Billing Adjudication Data (All Currently Displayed Rows)"
            else:  # No rows selected, use all from table_inv_data_full
                temp_billing_df = df_inv_full_for_report.copy()
                billing_table_caption_text = "Billing Adjudication Data (All Currently Displayed Rows)"

            if not temp_billing_df.empty:
                if 'Appearance' in temp_billing_df.columns:
                    df_billing_report_final_for_html = temp_billing_df.drop(columns=['Appearance'])
                else:
                    df_billing_report_final_for_html = temp_billing_df

    billing_table_html_content = df_to_html_table(df_billing_report_final_for_html, table_id="inv_table_report",
                                                  caption=billing_table_caption_text)

    viz_map_html = fig_to_base64_img_html(fig_viz_data, title="Callsign Path Map")
    viz_table_html = df_to_html_table(table_viz_data, table_id="viz_table_report", caption="Callsign Path Data")

    date_filter_info_html = ""
    if date_filter_summary_viz:
        summary_text = str(date_filter_summary_viz)
        if "Displaying:" in summary_text and "points)" in summary_text:
            date_filter_info_html = f"""
                <p style='text-align:left; font-style:italic; color:#333; margin-bottom:15px; font-size:0.9em;'>
                    <strong>Date Filter Applied:</strong> {summary_text}
                </p>"""

    report_html_content = f"""
        <!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Flight Operations Report</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; margin: 20px; background-color: #e9ecef; color: #333; font-size:14px; }}
            .container-report {{ max-width: 1140px; margin: auto; background-color: #ffffff; padding: 30px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
            .report-header {{ text-align: center; margin-bottom: 30px; padding-bottom: 20px; border-bottom: 2px solid #007bff; }}
            .logo {{ max-height: 60px; margin-bottom: 10px; }}
            .report-header h1 {{ color: #007bff; font-weight: 600; font-size: 2em; }}
            .address-block {{ font-size: 0.9em; color: #555; line-height: 1.5; }}
            .section {{ margin-bottom: 40px; padding: 25px; border: 1px solid #ddd; border-radius: 6px; background-color: #f8f9fa; }}
            .section h2 {{ color: #007bff; border-bottom: 1px solid #add8e6; padding-bottom: 10px; margin-bottom: 20px; font-size: 1.6em; font-weight: 500;}}
            .section h3 {{ color: #0056b3; font-size: 1.3em; margin-top: 20px; margin-bottom: 15px; font-weight: 500; }}
            table.table {{ width: 100%; margin-bottom: 1rem; color: #212529; border: 1px solid #dee2e6; }}
            table.table th, table.table td {{ padding: .6rem; vertical-align: top; border-top: 1px solid #dee2e6; font-size: 0.85rem; }}
            table.table thead th {{ vertical-align: bottom; border-bottom: 2px solid #dee2e6; background-color: #f1f1f1; color: #333; font-weight: 600;}}
            table.table-striped tbody tr:nth-of-type(odd) {{ background-color: rgba(0,0,0,.03); }}
            table.table-hover tbody tr:hover {{ color: #212529; background-color: rgba(0,0,0,.06); }}
            img.map-image {{ max-width: 100%; height: auto; display: block; margin: 10px auto; border: 1px solid #ccc; border-radius: 4px; }}
            .report-footer {{ display: flex; justify-content: space-between; align-items: flex-end; margin-top: 30px; padding-top: 20px; border-top: 1px solid #ccc; font-size: 0.85em; color: #666; }}
            .footer-left {{ text-align: left; }}
            .footer-center {{ text-align: center; }}
            .footer-right {{ text-align: right; }}
            .stamp-date {{ font-size: 0.95em; color: #444; margin-bottom: 5px; }}
            .stamp-logo {{ max-height: 50px; margin-bottom: 5px; display: block; }}
            .stamp-signature {{ font-family: 'Pacifico', cursive; font-size: 1.4em; color: #003366; margin-top: 8px; }}
            @media print {{ 
                body {{ margin: 0.5in; background-color: #fff; font-size: 12pt; }} 
                .container-report {{ box-shadow: none; border: none; padding: 0; width: 100%; max-width:100%;}} 
                .section {{ page-break-inside: avoid; border: none !important; background-color: #fff !important; box-shadow: none !important; padding: 15px 0;}} 
                .no-print {{ display: none !important; }} 
                table.table, img.map-image, .report-footer {{ page-break-inside: avoid; }} 
                .report-header h1, .section h2, .section h3 {{ color: #333 !important; }}
                table.table th {{ background-color: #eee !important; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
                .report-header {{ border-bottom: 2px solid #666 !important; }}
                .section h2 {{ border-bottom: 1px solid #ccc !important; }}
            }}
        </style>
        <link href="https://fonts.googleapis.com/css2?family=Pacifico&display=swap" rel="stylesheet"> </head><body><div class="container-report">
        <div class="report-header">
            {f'<img src="{report_logo_b64}" alt="Company Logo" class="logo">' if report_logo_b64 else '<p style="color:red;font-style:italic;">Logo not found.</p>'}
            <h1>Flight Operations Summary Report</h1>
            <p class="text-muted">Generated on: {datetime.now(DISPLAY_TIMEZONE).strftime('%A, %B %d, %Y at %H:%M:%S %Z')}</p>
        </div>

        <div class="section">
            <h2>Callsign Path Visualizer</h2>
            {date_filter_info_html}
            {viz_map_html}
            {viz_table_html}
        </div>

        <div class="section">
            <h2>Flight Investigation & Billing Tool</h2>
            {fig_to_base64_img_html(fig_inv_data, title="Billing Adjudication Map")}
            {billing_table_html_content} 
        </div>

        <div class="report-footer">
            <div class="footer-left">
                <div class="stamp-date">Date: {datetime.now(DISPLAY_TIMEZONE).strftime('%B %d, %Y')}</div>
                {f'<img src="{report_logo_b64}" alt="Logo" class="stamp-logo">' if report_logo_b64 else ''}
                <div class="stamp-signature">Operations Dept.</div>
            </div>
            <div class="footer-center address-block">
                <p><strong>South Sudan Civil Aviation Authority</strong><br>Kololo Road, PO Box 12345<br>Juba, South Sudan</p>
                <p>Tel: +211 955 119 177 | Fax +211 923 051 002 | Email: info@ssdcaa.gov.ss<br>Web: www.ssdcaa.gov.ss</p>
            </div>
            <div class="footer-right">
                 <p>Page <span class="page-number">1</span></p> </div>
        </div>
        </div></body></html>"""

    return dcc.send_string(report_html_content, "Flight_Operations_Report.html"), "Download Full Report (HTML)", False


if __name__ == "__main__":
    print(
        f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] Initializing Unified Multi-Tool Flight Dashboard (Vapour Theme)...");
    app.run(debug=True, port=5610)
