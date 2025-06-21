import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Set page configuration
st.set_page_config(
    page_title="Air Quality Analysis Dashboard",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define paths
DATA_PATH = "dashboard/air_quality_cleaned.csv"
VIS_PATH = "visualizations"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .section-header {
        font-size: 1.8rem;
        color: #1976D2;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .insight-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F0F2F6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Load data function
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH, index_col='datetime', parse_dates=True)
        return df
    except FileNotFoundError:
        st.error(f"Data file not found at {DATA_PATH}")
        return None

# Header section
def render_header():
    st.markdown('<h1 class="main-header">Air Quality Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("""
    This dashboard visualizes the analysis of air quality data from multiple monitoring stations. 
    The analysis focuses on two key business questions:
    1. **Temporal Trends**: How do air pollutant concentrations fluctuate over time?
    2. **Pollutant Relationships**: What correlations exist between different pollutants and meteorological factors?
    """)

# Temporal trends section
def render_temporal_trends(df):
    st.markdown('<h2 class="section-header">Temporal Trends Analysis</h2>', unsafe_allow_html=True)
    
    # Sidebar filters for temporal analysis
    st.sidebar.markdown("### Temporal Analysis Filters")
    
    # Get available pollutants
    pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    available_pollutants = [p for p in pollutants if p in df.columns]
    
    # Pollutant selection
    selected_pollutants = st.sidebar.multiselect(
        "Select Pollutants to Display",
        options=available_pollutants,
        default=available_pollutants[:3] if len(available_pollutants) >= 3 else available_pollutants
    )
    
    if not selected_pollutants:
        st.warning("Please select at least one pollutant to display.")
        return
    
    # Date range selection
    min_date = df.index.min().date()
    max_date = df.index.max().date()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Handle single date selection
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = date_range
        end_date = date_range
    
    # Filter data by date
    mask = (df.index.date >= start_date) & (df.index.date <= end_date)
    filtered_df = df.loc[mask]
    
    # Create tabs for different temporal views
    tabs = st.tabs(["Daily Trends", "Monthly Patterns", "Seasonal Analysis"])
    
    # Daily trends tab
    with tabs[0]:
        st.subheader("Daily Average Pollutant Concentrations")
        
        # Resample to daily frequency
        daily_mean = filtered_df[selected_pollutants].resample('D').mean()
        
        # Create plotly figure
        fig = px.line(
            daily_mean, 
            x=daily_mean.index, 
            y=selected_pollutants,
            labels={"value": "Concentration", "variable": "Pollutant", "datetime": "Date"},
            title="Daily Average Pollutant Concentrations",
            height=500
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Concentration (µg/m³)",
            legend_title="Pollutant",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display insights
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("**Key Insights:**")
        st.markdown("""
        - Daily pollutant concentrations show significant fluctuations over time
        - Short-term spikes may indicate specific pollution events or weather conditions
        - Weekend vs. weekday patterns may be visible in some pollutants due to traffic and industrial activity differences
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Monthly patterns tab
    with tabs[1]:
        st.subheader("Monthly Average Pollutant Concentrations")
        
        # Resample to monthly frequency
        monthly_mean = filtered_df[selected_pollutants].resample('ME').mean()
        
        # Create plotly figure
        fig = px.line(
            monthly_mean, 
            x=monthly_mean.index, 
            y=selected_pollutants,
            labels={"value": "Concentration", "variable": "Pollutant", "datetime": "Month"},
            title="Monthly Average Pollutant Concentrations",
            markers=True,
            height=500
        )
        
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Concentration (µg/m³)",
            legend_title="Pollutant",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display insights
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("**Key Insights:**")
        st.markdown("""
        - Monthly averages smooth out daily fluctuations and reveal longer-term trends
        - Seasonal patterns become more apparent at this time scale
        - Some pollutants may show annual cycles related to seasonal activities or weather patterns
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Seasonal analysis tab
    with tabs[2]:
        st.subheader("Seasonal Pollutant Patterns")
        
        # Add month number for seasonal grouping
        filtered_df_with_month = filtered_df.copy()
        filtered_df_with_month['month'] = filtered_df_with_month.index.month
        
        # Calculate seasonal averages (by month)
        seasonal_mean = filtered_df_with_month.groupby('month')[selected_pollutants].mean()
        
        # Create plotly figure
        fig = px.bar(
            seasonal_mean,
            x=seasonal_mean.index,
            y=selected_pollutants,
            labels={"value": "Concentration", "variable": "Pollutant", "month": "Month"},
            title="Average Pollutant Concentrations by Month",
            height=500,
            barmode='group'
        )
        
        # Update x-axis labels to month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(1, 13)),
                ticktext=month_names
            ),
            xaxis_title="Month",
            yaxis_title="Concentration (µg/m³)",
            legend_title="Pollutant",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display insights
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("**Key Insights:**")
        st.markdown("""
        - Particulate matter (PM2.5, PM10) often shows higher concentrations in winter months
        - Ozone (O3) typically peaks during summer months due to increased sunlight and temperature
        - Seasonal patterns may be influenced by weather conditions, heating/cooling needs, and agricultural activities
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# Pollutant relationships section
def render_pollutant_relationships(df):
    st.markdown('<h2 class="section-header">Pollutant Relationships Analysis</h2>', unsafe_allow_html=True)
    
    # Get available pollutants and meteorological factors
    pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    meteo_cols = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
    
    available_pollutants = [p for p in pollutants if p in df.columns]
    available_meteo = [m for m in meteo_cols if m in df.columns]
    
    # Create tabs for different relationship views
    tabs = st.tabs(["Correlation Analysis", "Pollutant Scatter Plots", "Meteorological Factors"])
    
    # Correlation analysis tab
    with tabs[0]:
        st.subheader("Correlation Matrix of Pollutants and Meteorological Factors")
        
        # Sidebar filters for correlation analysis
        st.sidebar.markdown("### Correlation Analysis Filters")
        
        # Variables selection for correlation
        cols_for_corr = st.sidebar.multiselect(
            "Select Variables for Correlation Analysis",
            options=available_pollutants + available_meteo,
            default=available_pollutants
        )
        
        if not cols_for_corr or len(cols_for_corr) < 2:
            st.warning("Please select at least two variables for correlation analysis.")
        else:
            # Calculate correlation matrix
            correlation_matrix = df[cols_for_corr].corr()
            
            # Create heatmap using plotly
            fig = px.imshow(
                correlation_matrix,
                text_auto='.2f',
                color_continuous_scale='RdBu_r',
                labels=dict(x="Variable", y="Variable", color="Correlation"),
                title="Correlation Matrix of Selected Variables"
            )
            
            fig.update_layout(
                height=600,
                width=800
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display insights
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.markdown("**Key Insights:**")
            st.markdown("""
            - Strong positive correlation between PM2.5 and PM10 (typically > 0.8)
            - NO2 and CO often show moderate positive correlation due to shared sources (traffic emissions)
            - Ozone (O3) typically shows negative correlation with NO2 due to atmospheric chemistry
            - Temperature often shows negative correlation with particulate matter
            """)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Pollutant scatter plots tab
    with tabs[1]:
        st.subheader("Relationships Between Pollutants")
        
        # Sidebar filters for scatter plots
        st.sidebar.markdown("### Scatter Plot Filters")
        
        # Variable selection for x and y axes
        x_var = st.sidebar.selectbox(
            "Select X-axis Variable",
            options=available_pollutants,
            index=1 if len(available_pollutants) > 1 else 0  # Default to PM10 if available
        )
        
        y_var = st.sidebar.selectbox(
            "Select Y-axis Variable",
            options=available_pollutants,
            index=0  # Default to PM2.5
        )
        
        # Sample size slider to prevent overplotting
        sample_size = st.sidebar.slider(
            "Sample Size",
            min_value=100,
            max_value=min(10000, len(df)),
            value=min(5000, len(df)),
            step=100
        )
        
        # Add trend line option
        add_trendline = st.sidebar.checkbox("Add Trend Line", value=True)
        
        # Create scatter plot
        if x_var != y_var:
            # Sample data to prevent overplotting
            sampled_df = df.sample(n=sample_size, random_state=42)
            
            # Create plotly figure
            fig = px.scatter(
                sampled_df,
                x=x_var,
                y=y_var,
                opacity=0.6,
                title=f"Relationship between {x_var} and {y_var}",
                trendline='ols' if add_trendline else None,
                height=600
            )
            
            fig.update_layout(
                xaxis_title=f"{x_var} Concentration (µg/m³)",
                yaxis_title=f"{y_var} Concentration (µg/m³)",
                hovermode="closest"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate and display correlation coefficient
            corr_coef = df[[x_var, y_var]].corr().iloc[0, 1]
            st.metric("Correlation Coefficient", f"{corr_coef:.3f}")
            
            # Display insights based on correlation strength
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.markdown("**Relationship Analysis:**")
            
            if abs(corr_coef) > 0.7:
                st.markdown(f"There is a **strong {'positive' if corr_coef > 0 else 'negative'}** correlation between {x_var} and {y_var}.")
            elif abs(corr_coef) > 0.3:
                st.markdown(f"There is a **moderate {'positive' if corr_coef > 0 else 'negative'}** correlation between {x_var} and {y_var}.")
            else:
                st.markdown(f"There is a **weak {'positive' if corr_coef > 0 else 'negative'}** correlation between {x_var} and {y_var}.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("Please select different variables for X and Y axes.")
    
    # Meteorological factors tab
    with tabs[2]:
        st.subheader("Pollutant and Meteorological Factor Relationships")
        
        if not available_meteo:
            st.info("No meteorological data available in the dataset.")
        else:
            # Sidebar filters for meteorological analysis
            st.sidebar.markdown("### Meteorological Analysis Filters")
            
            # Variable selection
            meteo_var = st.sidebar.selectbox(
                "Select Meteorological Factor",
                options=available_meteo,
                index=0  # Default to first available
            )
            
            pollutant_var = st.sidebar.selectbox(
                "Select Pollutant",
                options=available_pollutants,
                index=0  # Default to PM2.5
            )
            
            # Sample size slider
            sample_size = st.sidebar.slider(
                "Sample Size (Meteo)",
                min_value=100,
                max_value=min(10000, len(df)),
                value=min(5000, len(df)),
                step=100
            )
            
            # Add trend line option
            add_trendline = st.sidebar.checkbox("Add Trend Line (Meteo)", value=True)
            
            # Create scatter plot
            sampled_df = df.sample(n=sample_size, random_state=42)
            
            # Create plotly figure
            fig = px.scatter(
                sampled_df,
                x=meteo_var,
                y=pollutant_var,
                opacity=0.6,
                title=f"Relationship between {meteo_var} and {pollutant_var}",
                trendline='ols' if add_trendline else None,
                height=600
            )
            
            # Set appropriate axis labels based on meteorological variable
            x_label = meteo_var
            if meteo_var == 'TEMP':
                x_label += " (°C)"
            elif meteo_var == 'PRES':
                x_label += " (hPa)"
            elif meteo_var == 'WSPM':
                x_label += " (m/s)"
            
            fig.update_layout(
                xaxis_title=x_label,
                yaxis_title=f"{pollutant_var} Concentration (µg/m³)",
                hovermode="closest"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate and display correlation coefficient
            corr_coef = df[[meteo_var, pollutant_var]].corr().iloc[0, 1]
            st.metric("Correlation Coefficient", f"{corr_coef:.3f}")
            
            # Display insights
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.markdown("**Meteorological Relationship Analysis:**")
            
            # Custom insights based on meteorological variable
            if meteo_var == 'TEMP':
                st.markdown(f"""
                Temperature shows a {'positive' if corr_coef > 0 else 'negative'} correlation with {pollutant_var} 
                (coefficient: {corr_coef:.3f}). {'Higher' if corr_coef > 0 else 'Lower'} temperatures tend to be 
                associated with {'higher' if corr_coef > 0 else 'lower'} {pollutant_var} concentrations.
                
                This may be related to:
                - {'Increased photochemical reactions in warmer conditions' if corr_coef > 0 else 'Thermal inversions trapping pollutants in colder conditions'}
                - {'Enhanced evaporation of volatile compounds' if corr_coef > 0 else 'Increased emissions from heating sources in colder weather'}
                """)
            elif meteo_var == 'WSPM':
                st.markdown(f"""
                Wind speed shows a {'positive' if corr_coef > 0 else 'negative'} correlation with {pollutant_var} 
                (coefficient: {corr_coef:.3f}). {'Higher' if corr_coef > 0 else 'Lower'} wind speeds tend to be 
                associated with {'higher' if corr_coef > 0 else 'lower'} {pollutant_var} concentrations.
                
                This is likely due to:
                - {'Transport of pollutants from other regions' if corr_coef > 0 else 'Dispersion of local pollutants in higher wind conditions'}
                - {'Resuspension of particulate matter in windy conditions' if corr_coef > 0 and 'PM' in pollutant_var else 'Stagnant air allowing pollutant buildup in low wind conditions'}
                """)
            else:
                st.markdown(f"""
                {meteo_var} shows a {'positive' if corr_coef > 0 else 'negative'} correlation with {pollutant_var} 
                (coefficient: {corr_coef:.3f}). {'Higher' if corr_coef > 0 else 'Lower'} {meteo_var} values tend to be 
                associated with {'higher' if corr_coef > 0 else 'lower'} {pollutant_var} concentrations.
                """)
            
            st.markdown('</div>', unsafe_allow_html=True)

# Station comparison section
def render_station_comparison(df):
    if 'station' not in df.columns:
        st.info("Station information not available in the dataset.")
        return
        
    st.markdown('<h2 class="section-header">Station Comparison</h2>', unsafe_allow_html=True)
    
    # Get available stations and pollutants
    stations = df['station'].unique()
    pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    available_pollutants = [p for p in pollutants if p in df.columns]
    
    # Sidebar filters for station comparison
    st.sidebar.markdown("### Station Comparison Filters")
    
    # Pollutant selection
    selected_pollutant = st.sidebar.selectbox(
        "Select Pollutant for Station Comparison",
        options=available_pollutants,
        index=0  # Default to PM2.5
    )
    
    # Calculate station averages
    station_avg = df.groupby('station')[selected_pollutant].mean().sort_values(ascending=False)
    
    # Create bar chart
    fig = px.bar(
        x=station_avg.index,
        y=station_avg.values,
        labels={"x": "Station", "y": f"{selected_pollutant} Concentration (µg/m³)"},
        title=f"Average {selected_pollutant} Concentration by Station",
        height=500
    )
    
    fig.update_layout(
        xaxis_title="Station",
        yaxis_title=f"{selected_pollutant} Concentration (µg/m³)"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display insights
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("**Station Comparison Insights:**")
    st.markdown(f"""
    - Station {station_avg.index[0]} has the highest average {selected_pollutant} concentration ({station_avg.values[0]:.2f} µg/m³)
    - Station {station_avg.index[-1]} has the lowest average {selected_pollutant} concentration ({station_avg.values[-1]:.2f} µg/m³)
    - The difference between highest and lowest station is {(station_avg.values[0] - station_avg.values[-1]):.2f} µg/m³
    - These differences may be related to local emission sources, population density, or industrial activity near stations
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Station profiles
    st.subheader("Station Profiles")
    
    # Select stations to compare
    selected_stations = st.multiselect(
        "Select Stations to Compare",
        options=stations,
        default=stations[:min(3, len(stations))]
    )
    
    if selected_stations:
        # Filter data for selected stations
        station_data = df[df['station'].isin(selected_stations)]
        
        # Calculate daily averages for each station
        daily_station_avg = station_data.groupby(['station', pd.Grouper(freq='D')])[selected_pollutant].mean().reset_index()
        
        # Create line chart
        fig = px.line(
            daily_station_avg,
            x='datetime',
            y=selected_pollutant,
            color='station',
            labels={"datetime": "Date", selected_pollutant: f"{selected_pollutant} Concentration (µg/m³)"},
            title=f"Daily {selected_pollutant} Concentration by Station",
            height=500
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title=f"{selected_pollutant} Concentration (µg/m³)",
            legend_title="Station",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)

# About section
def render_about():
    st.markdown('<h2 class="section-header">About This Dashboard</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Data Source
    This dashboard uses air quality data from multiple monitoring stations. The dataset includes measurements of various pollutants:
    - **PM2.5**: Fine particulate matter with diameter less than 2.5 micrometers
    - **PM10**: Particulate matter with diameter less than 10 micrometers
    - **SO2**: Sulfur dioxide
    - **NO2**: Nitrogen dioxide
    - **CO**: Carbon monoxide
    - **O3**: Ozone
    
    ### Methodology
    The analysis follows these steps:
    1. Data cleaning and preprocessing (handling missing values, combining date/time columns)
    2. Temporal analysis (daily, monthly, and seasonal patterns)
    3. Correlation analysis between pollutants and meteorological factors
    4. Station comparison
    
    ### References
    - World Health Organization (WHO) Air Quality Guidelines
    - Original data source: [HTI GitHub Repository](https://github.com/marceloreis/HTI.git)
    
    ### Tools Used
    - Python for data processing
    - Pandas for data manipulation
    - Plotly for interactive visualizations
    - Streamlit for dashboard framework
    """)

# Main function
def main():
    # Load data
    df = load_data()
    
    if df is None:
        st.error("Failed to load data. Please check the data file path.")
        return
    
    # Render header
    render_header()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Temporal Trends", "Pollutant Relationships", "Station Comparison", "About"]
    )
    
    # Render selected page
    if page == "Temporal Trends":
        render_temporal_trends(df)
    elif page == "Pollutant Relationships":
        render_pollutant_relationships(df)
    elif page == "Station Comparison":
        render_station_comparison(df)
    else:
        render_about()
    
    # Footer
    st.markdown("---")
    st.markdown("Air Quality Analysis Dashboard | Created with Streamlit")

if __name__ == "__main__":
    main()
