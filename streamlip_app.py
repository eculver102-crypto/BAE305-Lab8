"""
Lab 10 - Water Quality Analysis Streamlit App
Combines station mapping and water quality trend analysis
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Water Quality Analysis App",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #34495e;
        padding: 0.5rem;
    }
    .info-text {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================
# DATA LOADING FUNCTIONS
# ============================================

@st.cache_data
def load_station_data(station_file):
    """Load and cache station data"""
    df = pd.read_csv(station_file)
    df = df.drop_duplicates(subset=['MonitoringLocationIdentifier'])
    df = df.dropna(subset=['LatitudeMeasure', 'LongitudeMeasure'])
    return df


@st.cache_data
def load_water_quality_data(water_file):
    """Load and clean water quality data"""
    df = pd.read_csv(water_file)
    
    # Clean the data
    df = df.dropna(subset=['CharacteristicName', 'ResultMeasureValue', 'ActivityStartDate', 'MonitoringLocationIdentifier'])
    df['ResultMeasureValue'] = pd.to_numeric(df['ResultMeasureValue'], errors='coerce')
    df = df.dropna(subset=['ResultMeasureValue'])
    
    # Remove zeros (but keep other values)
    df = df[df['ResultMeasureValue'] != 0]
    
    # Clean dates
    df['ActivityStartDate'] = pd.to_datetime(df['ActivityStartDate'], errors='coerce')
    df = df.dropna(subset=['ActivityStartDate'])
    
    return df


@st.cache_data
def get_station_info(stations_df):
    """Create station ID to name/organization mapping"""
    station_to_name = {}
    station_to_org = {}
    
    for idx, row in stations_df.iterrows():
        station_id = row['MonitoringLocationIdentifier']
        station_to_name[station_id] = row.get('MonitoringLocationName', station_id)
        org_name = row.get('OrganizationFormalName', None)
        if pd.isna(org_name):
            org_name = row.get('OrganizationIdentifier', 'Unknown')
        station_to_org[station_id] = org_name
    
    return station_to_name, station_to_org


# ============================================
# PART 1: MAP FUNCTIONS
# ============================================

def create_station_map(stations_df):
    """Create an interactive map with all stations"""
    
    # Calculate center
    center_lat = stations_df['LatitudeMeasure'].mean()
    center_lon = stations_df['LongitudeMeasure'].mean()
    
    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=8, control_scale=True)
    
    # Add tile layers
    folium.TileLayer('OpenStreetMap', name='Street').add_to(m)
    folium.TileLayer('CartoDB positron', name='Light').add_to(m)
    folium.TileLayer('CartoDB dark_matter', name='Dark').add_to(m)
    
    # Color mapping for station types
    type_colors = {
        'Stream': 'blue',
        'River/Stream': 'green',
        'Spring': 'orange',
        'Lake/Reservoir': 'purple',
        'Well': 'red'
    }
    
    # Add markers
    marker_cluster = MarkerCluster().add_to(m)
    
    for idx, row in stations_df.iterrows():
        station_type = row.get('MonitoringLocationTypeName', 'Unknown')
        if pd.isna(station_type):
            station_type = 'Unknown'
        color = type_colors.get(station_type, 'gray')
        
        station_name = row.get('MonitoringLocationName', row['MonitoringLocationIdentifier'])
        if pd.isna(station_name):
            station_name = row['MonitoringLocationIdentifier']
        
        org_name = row.get('OrganizationFormalName', row.get('OrganizationIdentifier', 'Unknown'))
        if pd.isna(org_name):
            org_name = 'Unknown'
        
        popup_html = f"""
        <div style="font-family: Arial; font-size: 13px; min-width: 220px;">
            <b>{station_name}</b><br>
            <b>ID:</b> {row['MonitoringLocationIdentifier']}<br>
            <b>Type:</b> {station_type}<br>
            <b>Organization:</b> {org_name}<br>
            <b>Lat:</b> {row['LatitudeMeasure']:.6f}<br>
            <b>Lon:</b> {row['LongitudeMeasure']:.6f}
        </div>
        """
        
        folium.Marker(
            location=[row['LatitudeMeasure'], row['LongitudeMeasure']],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=station_name,
            icon=folium.Icon(color=color, icon='tint', prefix='fa')
        ).add_to(marker_cluster)
    
    folium.LayerControl().add_to(m)
    
    return m


# ============================================
# PART 2: PLOTTING FUNCTIONS
# ============================================

def plot_characteristic_trend(water_df, station_to_name, station_to_org, 
                               characteristic, fraction_filter=None):
    """Create time series plot for a single characteristic"""
    
    # Filter data
    plot_df = water_df[water_df['CharacteristicName'] == characteristic].copy()
    
    if fraction_filter and fraction_filter != 'None':
        plot_df = plot_df[plot_df['ResultSampleFractionText'] == fraction_filter]
    
    if len(plot_df) == 0:
        st.warning(f"No data found for {characteristic}")
        return None
    
    # Get units
    units = plot_df['ResultMeasure/MeasureUnitCode'].iloc[0]
    if pd.isna(units):
        units = ''
    
    # Create plotly figure
    fig = go.Figure()
    
    # Add trace for each station
    stations = plot_df['MonitoringLocationIdentifier'].unique()
    
    for station in stations:
        station_data = plot_df[plot_df['MonitoringLocationIdentifier'] == station].sort_values('ActivityStartDate')
        
        if len(station_data) == 0:
            continue
        
        station_name = station_to_name.get(station, station)
        org_name = station_to_org.get(station, 'Unknown')
        label = f"{org_name[:25]} - {station_name[:25]}"
        
        fig.add_trace(go.Scatter(
            x=station_data['ActivityStartDate'],
            y=station_data['ResultMeasureValue'],
            mode='lines+markers',
            name=label,
            line=dict(width=2),
            marker=dict(size=6)
        ))
    
    # Update layout
    fraction_text = f" ({fraction_filter})" if fraction_filter and fraction_filter != 'None' else ""
    fig.update_layout(
        title=f'{characteristic}{fraction_text} - Time Series by Station',
        xaxis_title='Date',
        yaxis_title=f'Value ({units})',
        hovermode='closest',
        height=500,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05,
            font=dict(size=10)
        ),
        margin=dict(l=50, r=200, t=50, b=50)
    )
    
    return fig


def plot_two_characteristics(water_df, station_to_name, station_to_org,
                              characteristic1, characteristic2, fraction_filter=None):
    """Create comparison plot for two characteristics"""
    
    # Filter data for both characteristics
    df1 = water_df[water_df['CharacteristicName'] == characteristic1].copy()
    df2 = water_df[water_df['CharacteristicName'] == characteristic2].copy()
    
    if fraction_filter and fraction_filter != 'None':
        df1 = df1[df1['ResultSampleFractionText'] == fraction_filter]
        df2 = df2[df2['ResultSampleFractionText'] == fraction_filter]
    
    # Find common stations
    stations1 = set(df1['MonitoringLocationIdentifier'].unique())
    stations2 = set(df2['MonitoringLocationIdentifier'].unique())
    common_stations = stations1.intersection(stations2)
    
    if len(common_stations) == 0:
        st.warning(f"No stations have data for both {characteristic1} and {characteristic2}")
        return None
    
    # Get units
    units1 = df1['ResultMeasure/MeasureUnitCode'].iloc[0] if len(df1) > 0 else ''
    units2 = df2['ResultMeasure/MeasureUnitCode'].iloc[0] if len(df2) > 0 else ''
    
    # Create subplots
    n_stations = len(common_stations)
    n_cols = min(3, n_stations)
    n_rows = (n_stations + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_stations == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    fraction_text = f" ({fraction_filter})" if fraction_filter and fraction_filter != 'None' else ""
    fig.suptitle(f'{characteristic1} vs {characteristic2}{fraction_text}', fontsize=14, fontweight='bold')
    
    for idx, station in enumerate(sorted(common_stations)):
        ax = axes[idx]
        
        data1 = df1[df1['MonitoringLocationIdentifier'] == station].sort_values('ActivityStartDate')
        data2 = df2[df2['MonitoringLocationIdentifier'] == station].sort_values('ActivityStartDate')
        
        # Plot characteristic 1
        ax.plot(data1['ActivityStartDate'], data1['ResultMeasureValue'], 
                'b-o', label=characteristic1, linewidth=2, markersize=6)
        ax.set_xlabel('Date', fontsize=9)
        ax.set_ylabel(f'{characteristic1}\n({units1})', color='b', fontsize=9)
        ax.tick_params(axis='y', labelcolor='b')
        ax.grid(True, alpha=0.3)
        
        # Plot characteristic 2 on secondary axis
        ax2 = ax.twinx()
        ax2.plot(data2['ActivityStartDate'], data2['ResultMeasureValue'], 
                 'r-s', label=characteristic2, linewidth=2, markersize=6)
        ax2.set_ylabel(f'{characteristic2}\n({units2})', color='r', fontsize=9)
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Title
        station_name = station_to_name.get(station, station)
        org_name = station_to_org.get(station, 'Unknown')
        ax.set_title(f'{org_name[:20]}\n{station_name[:25]}', fontsize=9)
        
        ax.tick_params(axis='x', rotation=45, labelsize=8)
    
    # Hide unused subplots
    for idx in range(len(common_stations), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig


# ============================================
# MAIN APP
# ============================================

def main():
    # Header
    st.markdown('<div class="main-header">💧 Water Quality Analysis Dashboard</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar for file uploads
    with st.sidebar:
        st.markdown("## 📁 Data Upload")
        st.markdown("Please upload your data files:")
        
        station_file = st.file_uploader("Station Database (station.csv)", type=['csv'])
        water_file = st.file_uploader("Water Quality Database (narrowresult.csv)", type=['csv'])
        
        st.markdown("---")
        st.markdown("### 📊 About")
        st.markdown("""
        This app analyzes water quality data from monitoring stations.
        
        **Features:**
        - Interactive map of monitoring stations
        - Time series plots of water quality parameters
        - Compare two characteristics side-by-side
        """)
    
    # Check if files are uploaded
    if station_file is None or water_file is None:
        st.info("👈 Please upload both CSV files to begin analysis")
        
        # Show example of expected format
        with st.expander("📋 Expected File Format"):
            st.markdown("**station.csv** should contain:")
            st.code("MonitoringLocationIdentifier, MonitoringLocationName, MonitoringLocationTypeName, LatitudeMeasure, LongitudeMeasure, OrganizationFormalName")
            st.markdown("**narrowresult.csv** should contain:")
            st.code("CharacteristicName, ResultMeasureValue, ActivityStartDate, MonitoringLocationIdentifier, ResultMeasure/MeasureUnitCode, ResultSampleFractionText")
        return
    
    # Load data
    with st.spinner("Loading station data..."):
        stations_df = load_station_data(station_file)
        station_to_name, station_to_org = get_station_info(stations_df)
    
    with st.spinner("Loading water quality data..."):
        water_df = load_water_quality_data(water_file)
    
    st.success(f"✅ Loaded {len(stations_df)} stations and {len(water_df):,} water quality records")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🗺️ Station Map", "📈 Single Characteristic", "🔄 Compare Two Characteristics"])
    
    # ============================================
    # TAB 1: Station Map
    # ============================================
    with tab1:
        st.markdown('<div class="sub-header">🗺️ Water Quality Monitoring Stations</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.markdown("### 📊 Station Summary")
            st.metric("Total Stations", len(stations_df))
            
            # Station type breakdown
            type_counts = stations_df['MonitoringLocationTypeName'].value_counts()
            st.markdown("**Station Types:**")
            for st_type, count in type_counts.items():
                st.markdown(f"- {st_type}: {count}")
            
            # Organization breakdown
            org_counts = stations_df['OrganizationFormalName'].value_counts()
            st.markdown("**Organizations:**")
            for org, count in org_counts.head(5).items():
                st.markdown(f"- {org[:30]}: {count}")
        
        with col1:
            # Create and display map
            with st.spinner("Creating map..."):
                station_map = create_station_map(stations_df)
                st_folium(station_map, width=700, height=500)
    
    # ============================================
    # TAB 2: Single Characteristic
    # ============================================
    with tab2:
        st.markdown('<div class="sub-header">📈 Water Quality Trends</div>', unsafe_allow_html=True)
        
        # Get available characteristics
        characteristics = sorted(water_df['CharacteristicName'].unique())
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_char = st.selectbox("Select Characteristic", characteristics)
        
        with col2:
            fraction_options = ['None'] + sorted(water_df['ResultSampleFractionText'].dropna().unique().tolist())
            selected_fraction = st.selectbox("Fraction Filter (optional)", fraction_options)
        
        if selected_fraction == 'None':
            selected_fraction = None
        
        # Create and display plot
        if st.button("Generate Plot", key="single_plot"):
            with st.spinner("Creating plot..."):
                fig = plot_characteristic_trend(water_df, station_to_name, station_to_org,
                                                 selected_char, selected_fraction)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show data summary
                    with st.expander("📊 Data Summary"):
                        plot_df = water_df[water_df['CharacteristicName'] == selected_char].copy()
                        if selected_fraction:
                            plot_df = plot_df[plot_df['ResultSampleFractionText'] == selected_fraction]
                        
                        st.write(f"**Total records:** {len(plot_df)}")
                        st.write(f"**Number of stations:** {plot_df['MonitoringLocationIdentifier'].nunique()}")
                        st.write(f"**Date range:** {plot_df['ActivityStartDate'].min().date()} to {plot_df['ActivityStartDate'].max().date()}")
                        
                        # Station breakdown
                        st.write("**Station breakdown:**")
                        station_counts = plot_df['MonitoringLocationIdentifier'].value_counts()
                        for station, count in station_counts.head(10).items():
                            station_name = station_to_name.get(station, station)
                            org_name = station_to_org.get(station, 'Unknown')
                            st.write(f"- {org_name[:25]} - {station_name[:25]}: {count} samples")
    
    # ============================================
    # TAB 3: Compare Two Characteristics
    # ============================================
    with tab3:
        st.markdown('<div class="sub-header">🔄 Compare Two Characteristics</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            char1 = st.selectbox("First Characteristic", characteristics, key="char1")
        
        with col2:
            char2 = st.selectbox("Second Characteristic", characteristics, key="char2")
        
        with col3:
            fraction_options = ['None'] + sorted(water_df['ResultSampleFractionText'].dropna().unique().tolist())
            compare_fraction = st.selectbox("Fraction Filter", fraction_options, key="compare_frac")
        
        if compare_fraction == 'None':
            compare_fraction = None
        
        if st.button("Compare", key="compare_plot"):
            if char1 == char2:
                st.warning("Please select two different characteristics")
            else:
                with st.spinner("Creating comparison plot..."):
                    fig = plot_two_characteristics(water_df, station_to_name, station_to_org,
                                                    char1, char2, compare_fraction)
                    if fig:
                        st.pyplot(fig)
                        
                        # Show common stations
                        with st.expander("📊 Common Stations"):
                            df1 = water_df[water_df['CharacteristicName'] == char1].copy()
                            df2 = water_df[water_df['CharacteristicName'] == char2].copy()
                            
                            if compare_fraction:
                                df1 = df1[df1['ResultSampleFractionText'] == compare_fraction]
                                df2 = df2[df2['ResultSampleFractionText'] == compare_fraction]
                            
                            stations1 = set(df1['MonitoringLocationIdentifier'].unique())
                            stations2 = set(df2['MonitoringLocationIdentifier'].unique())
                            common = stations1.intersection(stations2)
                            
                            st.write(f"**Stations with both characteristics:** {len(common)}")
                            for station in sorted(common):
                                station_name = station_to_name.get(station, station)
                                org_name = station_to_org.get(station, 'Unknown')
                                st.write(f"- {org_name} - {station_name}")
    
    # Footer
    st.markdown("---")
    st.markdown("*Lab 10 - Python and AI: Water Quality Analysis*")


if __name__ == "__main__":
    main()
