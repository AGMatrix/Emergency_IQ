import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import kruskal, f_oneway
import sqlite3
import warnings
import os

warnings.filterwarnings('ignore')


# DATABASE CONNECTION AND LOADING FUNCTIONS
def connect_to_database(db_name='emergency_calls.db'):
    """
    Create connection to the SQLite database.

    Parameters:
    db_name (str): Name of the database file

    Returns:
    sqlite3.Connection: Database connection object
    """
    try:
        conn = sqlite3.connect(db_name)
        print(f"Connected to database: {db_name}")
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None


def load_and_prepare_data_from_db(db_name="emergency_calls.db"):
    """
    Load the 911 calls data from database with temporal features already prepared.

    Parameters:
    db_name (str): Path to the database file

    Returns:
    pandas.DataFrame: DataFrame with temporal features
    """
    print("Loading data from database...")

    conn = connect_to_database(db_name)
    if conn is None:
        return None

    try:
        # Load main data from database
        query = """
        SELECT call_id, lat, lng, description, zip_code, title, timestamp,
               township, address, emergency_code, hour, day_of_week,
               day_of_week_num, month, month_name, year, season, 
               category, time_period
        FROM emergency_calls
        ORDER BY timestamp
        """

        df = pd.read_sql_query(query, conn)

        # Convert timestamp back to datetime for analysis
        df['timeStamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timeStamp'].dt.date
        df['week_of_year'] = df['timeStamp'].dt.isocalendar().week

        # Rename columns to match original analysis code
        df = df.rename(columns={
            'zip_code': 'zip',
            'township': 'twp',
            'address': 'addr',
            'description': 'desc',
            'emergency_code': 'e'
        })

        print(f"Data loaded from database: {len(df):,} records")
        print(f"Date range: {df['timeStamp'].min()} to {df['timeStamp'].max()}")

        conn.close()
        return df

    except Exception as e:
        print(f"Error loading data from database: {e}")
        conn.close()
        return None


def get_summary_stats_from_db(db_name="emergency_calls.db"):
    """
    Get pre-computed summary statistics from database tables.

    Parameters:
    db_name (str): Database file name

    Returns:
    dict: Dictionary containing summary statistics
    """
    conn = connect_to_database(db_name)
    if conn is None:
        return {}

    try:
        # Get temporal patterns
        temporal_query = """
        SELECT pattern_type, pattern_value, call_count, percentage
        FROM temporal_patterns
        ORDER BY pattern_type, call_count DESC
        """
        temporal_df = pd.read_sql_query(temporal_query, conn)

        # Get ZIP code stats
        zip_query = """
        SELECT zip_code, total_calls, avg_calls_per_day, peak_hour, dominant_category
        FROM zip_code_stats
        ORDER BY total_calls DESC
        LIMIT 10
        """
        zip_df = pd.read_sql_query(zip_query, conn)

        # Get emergency type stats
        category_query = """
        SELECT category, total_calls, percentage, avg_calls_per_day, peak_hour
        FROM emergency_types
        ORDER BY total_calls DESC
        """
        category_df = pd.read_sql_query(category_query, conn)

        conn.close()

        return {
            'temporal_patterns': temporal_df,
            'zip_stats': zip_df,
            'category_stats': category_df
        }

    except Exception as e:
        print(f"Error getting summary stats: {e}")
        conn.close()
        return {}


def analyze_missing_zip_codes_db(db_name='emergency_calls.db'):
    """
    Analyze missing ZIP codes using database queries.

    Parameters:
    db_name (str): Path to the database file

    Returns:
    dict: Dictionary with analysis results
    """
    print("Analyzing missing ZIP codes from database...")

    conn = connect_to_database(db_name)
    if conn is None:
        return None

    try:
        # Get total record count
        total_query = "SELECT COUNT(*) FROM emergency_calls"
        total_records = pd.read_sql_query(total_query, conn).iloc[0, 0]

        # Count missing ZIP codes (NULL or empty)
        missing_query = """
        SELECT COUNT(*) FROM emergency_calls 
        WHERE zip_code IS NULL OR zip_code = '' OR zip_code = 'None'
        """
        total_missing = pd.read_sql_query(missing_query, conn).iloc[0, 0]

        # Valid ZIP codes
        valid_zip_codes = total_records - total_missing

        # Calculate percentages
        missing_percentage = (total_missing / total_records) * 100
        valid_percentage = (valid_zip_codes / total_records) * 100

        # Get sample of missing records
        sample_query = """
        SELECT township, address, title 
        FROM emergency_calls 
        WHERE zip_code IS NULL OR zip_code = '' OR zip_code = 'None'
        LIMIT 10
        """
        missing_sample = pd.read_sql_query(sample_query, conn)

        # Get records with coordinates
        coords_query = """
        SELECT COUNT(*) FROM emergency_calls 
        WHERE lat IS NOT NULL AND lng IS NOT NULL
        """
        coords_available = pd.read_sql_query(coords_query, conn).iloc[0, 0]

        # Missing ZIPs with coordinates
        missing_with_coords_query = """
        SELECT COUNT(*) FROM emergency_calls 
        WHERE (zip_code IS NULL OR zip_code = '' OR zip_code = 'None')
        AND lat IS NOT NULL AND lng IS NOT NULL
        """
        missing_with_coords = pd.read_sql_query(missing_with_coords_query, conn).iloc[0, 0]

        conn.close()

        # Analysis results
        results = {
            'total_records': total_records,
            'total_missing': total_missing,
            'valid_zip_codes': valid_zip_codes,
            'missing_percentage': missing_percentage,
            'valid_percentage': valid_percentage,
            'coords_available': coords_available,
            'missing_with_coords': missing_with_coords
        }

        # Print summary
        print("\n" + "=" * 60)
        print("ZIP CODE ANALYSIS SUMMARY (FROM DATABASE)")
        print("=" * 60)

        print(f"\nOVERALL STATISTICS:")
        print(f"   Total Records:           {int(total_records):8,d}")
        print(f"   Valid ZIP Codes:         {int(valid_zip_codes):8,d} ({valid_percentage:5.1f}%)")
        print(f"   Missing ZIP Codes:       {int(total_missing):8,d} ({missing_percentage:5.1f}%)")

        print(f"\nCOORDINATE INFORMATION:")
        print(f"   Records with coordinates: {int(coords_available):8,d}")
        print(f"   Missing ZIPs with coords: {int(missing_with_coords):8,d}")

        if missing_with_coords > 0:
            recovery_percentage = (missing_with_coords / total_missing) * 100
            print(f"   Potential recovery rate:  {recovery_percentage:7.1f}%")

        if len(missing_sample) > 0:
            print(f"\nSAMPLE RECORDS WITH MISSING ZIP CODES:")
            print("-" * 80)
            print(f"{'Township':<20} {'Address':<35} {'Title':<25}")
            print("-" * 80)

            for _, row in missing_sample.iterrows():
                township = str(row['township'])[:19] if pd.notna(row['township']) else 'Unknown'
                address = str(row['address'])[:34] if pd.notna(row['address']) else 'Unknown'
                title = str(row['title'])[:24] if pd.notna(row['title']) else 'Unknown'
                print(f"{township:<20} {address:<35} {title:<25}")

        return results

    except Exception as e:
        print(f"Error analyzing missing ZIP codes: {e}")
        conn.close()
        return None


def analyze_hourly_patterns_db(db_name="emergency_calls.db"):
    """
    Analyze call volume patterns by hour using database queries.

    Parameters:
    db_name (str): Database file name

    Returns:
    dict: Analysis results
    """
    print("\nANALYZING HOURLY PATTERNS (FROM DATABASE)")
    print("=" * 45)

    conn = connect_to_database(db_name)
    if conn is None:
        return {}

    try:
        # Get hourly patterns from pre-computed table
        hourly_query = """
        SELECT pattern_value, call_count, percentage
        FROM temporal_patterns 
        WHERE pattern_type = 'hourly'
        ORDER BY CAST(pattern_value AS INTEGER)
        """
        hourly_data = pd.read_sql_query(hourly_query, conn)

        # Convert to series for compatibility with existing code
        hourly_counts = pd.Series(
            hourly_data['call_count'].values,
            index=hourly_data['pattern_value'].astype(int)
        )

        # Peak and quiet hours
        peak_hours = hourly_counts.nlargest(5)
        quiet_hours = hourly_counts.nsmallest(5)

        # Evening hours analysis
        evening_hours = [17, 18, 19]
        evening_query = """
        SELECT SUM(call_count) FROM temporal_patterns
        WHERE pattern_type = 'hourly' AND pattern_value IN ('17', '18', '19')
        """
        evening_count = pd.read_sql_query(evening_query, conn).iloc[0, 0]

        total_query = "SELECT COUNT(*) FROM emergency_calls"
        total_calls = pd.read_sql_query(total_query, conn).iloc[0, 0]
        evening_percentage = (evening_count / total_calls) * 100

        # Peak hours by category
        category_peak_query = """
        SELECT category, hour, COUNT(*) as count
        FROM emergency_calls 
        WHERE category IN ('EMS', 'Fire', 'Traffic')
        GROUP BY category, hour
        ORDER BY category, count DESC
        """
        category_peaks = pd.read_sql_query(category_peak_query, conn)

        conn.close()

        print(f"PEAK HOURS (Highest call volumes):")
        for hour, count in peak_hours.items():
            percentage = (count / total_calls) * 100
            print(f"  {hour:2d}:00-{hour:2d}:59: {int(count):5,d} calls ({percentage:4.1f}%)")

        print(f"\nQUIET HOURS (Lowest call volumes):")
        for hour, count in quiet_hours.items():
            percentage = (count / total_calls) * 100
            print(f"  {hour:2d}:00-{hour:2d}:59: {int(count):5,d} calls ({percentage:4.1f}%)")

        print(f"\nEVENING HOURS ANALYSIS (17:00-19:59):")
        print(f"  Total evening calls: {int(evening_count):,d}")
        print(f"  Percentage of all calls: {evening_percentage:.1f}%")

        print(f"\nHOURLY PATTERNS BY EMERGENCY TYPE:")
        for category in ['EMS', 'Fire', 'Traffic']:
            cat_data = category_peaks[category_peaks['category'] == category]
            if len(cat_data) > 0:
                peak_hour = cat_data.iloc[0]['hour']
                peak_count = cat_data.iloc[0]['count']
                print(f"  {category:<8}: Peak at {peak_hour:2d}:00 ({int(peak_count):,d} calls)")

        return {
            'hourly_counts': hourly_counts,
            'peak_hours': peak_hours,
            'quiet_hours': quiet_hours,
            'evening_calls': evening_count,
            'evening_percentage': evening_percentage
        }

    except Exception as e:
        print(f"Error analyzing hourly patterns: {e}")
        conn.close()
        return {}


def analyze_missing_values_db(db_name="emergency_calls.db"):
    """Analyze missing values using database queries."""
    print("\nMISSING VALUE ANALYSIS (FROM DATABASE)")
    print("=" * 35)

    conn = connect_to_database(db_name)
    if conn is None:
        return {}

    try:
        # Check missing values for key columns
        missing_analysis = {}

        columns_to_check = ['zip_code', 'township', 'address', 'lat', 'lng', 'title']

        for col in columns_to_check:
            query = f"""
            SELECT 
                COUNT(*) as total,
                COUNT({col}) as non_null,
                COUNT(*) - COUNT({col}) as missing
            FROM emergency_calls
            """
            result = pd.read_sql_query(query, conn)

            total = result['total'].iloc[0]
            missing = result['missing'].iloc[0]

            if missing > 0:
                percentage = (missing / total) * 100
                missing_analysis[col] = {'count': missing, 'percentage': percentage}
                print(f"  {col:<20}: {missing:6,d} missing ({percentage:5.1f}%)")

        if not missing_analysis:
            print("No significant missing values detected.")

        conn.close()
        return missing_analysis

    except Exception as e:
        print(f"Error analyzing missing values: {e}")
        conn.close()
        return {}


def detect_outliers_db(db_name="emergency_calls.db"):
    """Detect outliers in daily call volumes using database."""
    print("\nOUTLIER DETECTION: Daily Call Volumes (FROM DATABASE)")
    print("=" * 50)

    conn = connect_to_database(db_name)
    if conn is None:
        return []

    try:
        # Get daily call counts
        daily_query = """
        SELECT DATE(timestamp) as call_date, COUNT(*) as daily_count
        FROM emergency_calls
        GROUP BY DATE(timestamp)
        ORDER BY call_date
        """
        daily_data = pd.read_sql_query(daily_query, conn)
        daily_counts = daily_data['daily_count']

        # Calculate quartiles and IQR
        q1 = daily_counts.quantile(0.25)
        q3 = daily_counts.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        # Find outliers
        outliers = daily_data[(daily_counts < lower) | (daily_counts > upper)]

        print(f"Found {len(outliers)} outlier days (out of {len(daily_data)} days)")
        print(f"Normal range: {lower:.1f} - {upper:.1f} calls per day")
        print(f"Q1: {q1:.1f}, Q3: {q3:.1f}, IQR: {iqr:.1f}")

        if len(outliers) > 0:
            print(f"\nOutlier days:")
            for _, row in outliers.head(10).iterrows():
                print(f"  {row['call_date']}: {row['daily_count']} calls")

        conn.close()
        return outliers

    except Exception as e:
        print(f"Error detecting outliers: {e}")
        conn.close()
        return []


def analyze_geospatial_db(db_name="emergency_calls.db"):
    """Analyze geographical distribution using database."""
    print("\nGEOSPATIAL ANALYSIS (FROM DATABASE)")
    print("=" * 35)

    conn = connect_to_database(db_name)
    if conn is None:
        return

    try:
        # Get ZIP code analysis from pre-computed table
        zip_query = """
        SELECT zip_code, total_calls, dominant_category
        FROM zip_code_stats
        ORDER BY total_calls DESC
        LIMIT 10
        """
        zip_data = pd.read_sql_query(zip_query, conn)

        # Get total calls for percentage calculation
        total_query = "SELECT COUNT(*) FROM emergency_calls"
        total_calls = pd.read_sql_query(total_query, conn).iloc[0, 0]

        print("Top 10 ZIP Codes by Call Volume:")
        for _, row in zip_data.iterrows():
            percentage = (row['total_calls'] / total_calls) * 100
            print(
                f"  {row['zip_code']}: {row['total_calls']:,d} calls ({percentage:.1f}%) - {row['dominant_category']}")

        # Township analysis
        township_query = """
        SELECT township, COUNT(*) as call_count
        FROM emergency_calls
        WHERE township IS NOT NULL AND township != 'Unknown'
        GROUP BY township
        ORDER BY call_count DESC
        LIMIT 5
        """
        township_data = pd.read_sql_query(township_query, conn)

        print(f"\nTop 5 Townships by Call Volume:")
        for _, row in township_data.iterrows():
            percentage = (row['call_count'] / total_calls) * 100
            print(f"  {row['township']}: {row['call_count']:,d} calls ({percentage:.1f}%)")

        # Coordinate availability
        coords_query = """
        SELECT 
            COUNT(*) as total,
            COUNT(lat) as with_lat,
            COUNT(lng) as with_lng,
            COUNT(CASE WHEN lat IS NOT NULL AND lng IS NOT NULL THEN 1 END) as with_both
        FROM emergency_calls
        """
        coords_data = pd.read_sql_query(coords_query, conn)

        coords_available = coords_data['with_both'].iloc[0]
        coords_percentage = (coords_available / total_calls) * 100

        print(f"\nCoordinate Information:")
        print(f"  Records with coordinates: {coords_available:,d} ({coords_percentage:.1f}%)")

        if coords_available > 0:
            # Basic coordinate statistics
            coord_stats_query = """
            SELECT 
                MIN(lat) as min_lat, MAX(lat) as max_lat,
                MIN(lng) as min_lng, MAX(lng) as max_lng
            FROM emergency_calls
            WHERE lat IS NOT NULL AND lng IS NOT NULL
            """
            coord_stats = pd.read_sql_query(coord_stats_query, conn)

            lat_range = coord_stats['max_lat'].iloc[0] - coord_stats['min_lat'].iloc[0]
            lng_range = coord_stats['max_lng'].iloc[0] - coord_stats['min_lng'].iloc[0]

            print(f"  Latitude range: {lat_range:.4f} degrees")
            print(f"  Longitude range: {lng_range:.4f} degrees")

        conn.close()

    except Exception as e:
        print(f"Error in geospatial analysis: {e}")
        conn.close()


# MODIFIED ANALYSIS FUNCTIONS TO USE DATABASE DATA
def analyze_daily_patterns(df):
    """
    Analyze call volume patterns by day of week.
    (Modified to work with database-loaded data)
    """
    print("\nANALYZING DAILY PATTERNS")
    print("=" * 40)

    # Day of week analysis
    daily_counts = df['day_of_week'].value_counts()

    # Reorder days properly
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_counts_ordered = daily_counts.reindex(day_order)

    total_calls = len(df)

    print("CALL VOLUME BY DAY OF WEEK:")
    for day, count in daily_counts_ordered.items():
        if pd.notna(count):
            percentage = (count / total_calls) * 100
            print(f"  {day:<10}: {int(count):6,d} calls ({percentage:4.1f}%)")
        else:
            print(f"  {day:<10}: {'N/A':>6} calls (  N/A%)")

    # Weekday vs Weekend analysis
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    weekends = ['Saturday', 'Sunday']

    weekday_calls = df[df['day_of_week'].isin(weekdays)]
    weekend_calls = df[df['day_of_week'].isin(weekends)]

    weekday_count = len(weekday_calls)
    weekend_count = len(weekend_calls)

    print(f"\nWEEKDAY vs WEEKEND ANALYSIS:")
    print(f"  Weekday calls: {int(weekday_count):,d} ({weekday_count / total_calls * 100:.1f}%)")
    print(f"  Weekend calls: {int(weekend_count):,d} ({weekend_count / total_calls * 100:.1f}%)")
    print(f"  Weekday average: {weekday_count / 5:,.0f} calls per day")
    print(f"  Weekend average: {weekend_count / 2:,.0f} calls per day")

    return {
        'daily_counts': daily_counts_ordered,
        'weekday_count': weekday_count,
        'weekend_count': weekend_count
    }


def analyze_monthly_seasonal_patterns(df):
    """
    Analyze call volume patterns by month and season.
    (Modified to work with database-loaded data)
    """
    print("\nANALYZING MONTHLY AND SEASONAL PATTERNS")
    print("=" * 50)

    # Monthly analysis
    monthly_counts = df['month_name'].value_counts()
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    monthly_counts_ordered = monthly_counts.reindex(month_order)

    total_calls = len(df)

    print(f"CALL VOLUME BY MONTH:")
    for month, count in monthly_counts_ordered.items():
        if pd.notna(count):
            percentage = (count / total_calls) * 100
            print(f"  {month:<10}: {int(count):6,d} calls ({percentage:4.1f}%)")

    # Seasonal analysis
    seasonal_counts = df['season'].value_counts()
    season_order = ['Spring', 'Summer', 'Fall', 'Winter']
    seasonal_counts_ordered = seasonal_counts.reindex(season_order)

    print(f"\nCALL VOLUME BY SEASON:")
    for season, count in seasonal_counts_ordered.items():
        if pd.notna(count):
            percentage = (count / total_calls) * 100
            print(f"  {season:<8}: {int(count):6,d} calls ({percentage:4.1f}%)")

    return {
        'monthly_counts': monthly_counts_ordered,
        'seasonal_counts': seasonal_counts_ordered
    }


def analyze_time_periods(df):
    """
    Analyze call volume patterns by time periods.
    (Modified to work with database-loaded data)
    """
    print("\nANALYZING TIME PERIOD PATTERNS")
    print("=" * 40)

    period_counts = df['time_period'].value_counts()
    period_order = ['Morning (6-12)', 'Afternoon (12-17)', 'Evening (17-21)', 'Night (21-6)']
    period_counts_ordered = period_counts.reindex(period_order)

    total_calls = len(df)

    print(f"CALL VOLUME BY TIME PERIOD:")
    for period, count in period_counts_ordered.items():
        if pd.notna(count):
            percentage = (count / total_calls) * 100
            print(f"  {period:<18}: {int(count):6,d} calls ({percentage:4.1f}%)")

    return {'period_counts': period_counts_ordered}


def create_visualizations(df, save_plots=True):
    """
    Create temporal analysis visualizations.
    (Modified to work with database-loaded data)
    """
    print("\nCREATING VISUALIZATIONS")
    print("=" * 30)

    # Set style
    plt.style.use('default')
    fig = plt.figure(figsize=(16, 12))

    # 1. Hourly distribution
    plt.subplot(2, 3, 1)
    hourly_counts = df['hour'].value_counts().sort_index()
    hourly_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Call Volume by Hour of Day')
    plt.xlabel('Hour')
    plt.ylabel('Number of Calls')
    plt.xticks(rotation=45)

    # Highlight evening hours (17-19)
    for i, hour in enumerate(hourly_counts.index):
        if hour in [17, 18, 19]:
            plt.bar(i, hourly_counts[hour], color='orange', edgecolor='black')

    # 2. Daily distribution
    plt.subplot(2, 3, 2)
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_counts = df['day_of_week'].value_counts().reindex(day_order)
    daily_counts.plot(kind='bar', color='lightgreen', edgecolor='black')
    plt.title('Call Volume by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Number of Calls')
    plt.xticks(rotation=45)

    # 3. Monthly distribution
    plt.subplot(2, 3, 3)
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    monthly_counts = df['month_name'].value_counts().reindex(month_order)
    monthly_counts.plot(kind='bar', color='lightcoral', edgecolor='black')
    plt.title('Call Volume by Month')
    plt.xlabel('Month')
    plt.ylabel('Number of Calls')
    plt.xticks(rotation=45)

    # 4. Time period distribution
    plt.subplot(2, 3, 4)
    period_order = ['Morning (6-12)', 'Afternoon (12-17)', 'Evening (17-21)', 'Night (21-6)']
    period_counts = df['time_period'].value_counts().reindex(period_order)
    colors = ['gold', 'orange', 'red', 'navy']
    period_counts.plot(kind='bar', color=colors, edgecolor='black')
    plt.title('Call Volume by Time Period')
    plt.xlabel('Time Period')
    plt.ylabel('Number of Calls')
    plt.xticks(rotation=45)

    # 5. Seasonal distribution
    plt.subplot(2, 3, 5)
    season_order = ['Spring', 'Summer', 'Fall', 'Winter']
    seasonal_counts = df['season'].value_counts().reindex(season_order)
    seasonal_colors = ['lightgreen', 'gold', 'orange', 'lightblue']
    seasonal_counts.plot(kind='bar', color=seasonal_colors, edgecolor='black')
    plt.title('Call Volume by Season')
    plt.xlabel('Season')
    plt.ylabel('Number of Calls')
    plt.xticks(rotation=0)

    # 6. Category distribution by hour (heatmap style)
    plt.subplot(2, 3, 6)
    category_hour = pd.crosstab(df['category'], df['hour'])

    # Simple visualization since seaborn might not be available
    category_hour_pct = category_hour.div(category_hour.sum(axis=1), axis=0) * 100

    # Plot each category
    categories = ['EMS', 'Fire', 'Traffic']
    colors_cat = ['red', 'orange', 'blue']

    for i, (cat, color) in enumerate(zip(categories, colors_cat)):
        if cat in category_hour_pct.index:
            plt.plot(range(24), category_hour_pct.loc[cat],
                     label=cat, color=color, linewidth=2, marker='o', markersize=3)

    plt.title('Emergency Type Distribution by Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Percentage of Category Calls')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plots:
        plt.savefig('temporal_analysis_911_calls_from_db.png', dpi=300, bbox_inches='tight')
        print("Visualizations saved as 'temporal_analysis_911_calls_from_db.png'")

    plt.show()


def bivariate_analysis(df):
    """Analyze relationships between categorical variables."""
    print("\nBIVARIATE ANALYSIS: Category vs Day of Week")
    print("=" * 45)

    ct = pd.crosstab(df['day_of_week'], df['category'], normalize='index') * 100
    print("Percentage distribution by day of week:")
    print(ct.round(1).to_string())

    # Create visualization
    plt.figure(figsize=(12, 6))
    ct.plot(kind='bar', stacked=True, colormap='Set2', edgecolor='black')
    plt.title("Emergency Category Distribution by Day of Week")
    plt.ylabel("Percentage")
    plt.xlabel("Day of Week")
    plt.xticks(rotation=45)
    plt.legend(title="Category", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("bivariate_category_day_from_db.png", dpi=300, bbox_inches='tight')
    plt.show()


def correlation_heatmap(df):
    """Plot a correlation heatmap of numeric features."""
    print("\nCORRELATION HEATMAP")
    print("=" * 30)

    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty or numeric_df.shape[1] < 2:
        print("Insufficient numeric features for correlation analysis.")
        return

    # Remove constant columns
    numeric_df = numeric_df.loc[:, numeric_df.nunique() > 1]

    if numeric_df.shape[1] < 2:
        print("Insufficient varying numeric features for correlation analysis.")
        return

    corr = numeric_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0, square=True,
                cbar_kws={'shrink': 0.75})
    plt.title("Correlation Matrix of Numeric Features")
    plt.tight_layout()
    plt.savefig("correlation_heatmap_from_db.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_time_series(df):
    """Plot time series analysis for call volumes."""
    # Ensure timestamp is datetime
    df['timeStamp'] = pd.to_datetime(df['timeStamp'])

    # Filter for December
    december_df = df[df['timeStamp'].dt.month == 12]

    # Group by full date
    daily_counts = december_df.groupby(df['timeStamp'].dt.date).size()

    plt.figure(figsize=(12, 6))
    daily_counts.plot(marker='o', linestyle='-')
    plt.title("911 Call Volume Trend - December (From Database)")
    plt.xlabel("Date")
    plt.ylabel("Number of Calls")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def apply_pca(df, n_components=0.95):
    """Apply PCA for dimensionality reduction."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Clean constant/missing cols
    clean_cols = [col for col in numeric_cols if df[col].nunique() > 1 and df[col].notna().all()]
    X = df[clean_cols]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Create dataframe with PCA results
    pca_df = pd.DataFrame(X_pca, columns=[f"PC{i + 1}" for i in range(X_pca.shape[1])])

    print("\nAPPLYING PCA FOR DIMENSIONALITY REDUCTION")
    print("=" * 45)
    print(f"Numeric features used for PCA: {clean_cols}")
    print(f"Original feature count: {len(clean_cols)}")
    print(f"Reduced to {pca.n_components_} components (explained variance ≥ {n_components})")

    return pca_df, pca, clean_cols


def explain_pca_components(pca_model, feature_names):
    print("\nPCA COMPONENT LOADINGS (Feature Weights):")
    print("=" * 50)

    loadings = pd.DataFrame(
        pca_model.components_,
        columns=feature_names,
        index=[f'PC{i + 1}' for i in range(pca_model.n_components_)]
    )
    print(loadings.round(3))

    plt.figure(figsize=(10, 6))
    sns.heatmap(loadings, annot=True, cmap='coolwarm', center=0)
    plt.title("PCA Component Loadings")
    plt.xlabel("Original Features")
    plt.ylabel("Principal Components")
    plt.tight_layout()
    plt.savefig("pca_component_loadings_from_db.png")
    plt.show()

    # Explain components
    print("\nCOMPONENT INTERPRETATION:")
    for i, pc in enumerate(loadings.index):
        print(f"\n{pc} (Explains {pca_model.explained_variance_ratio_[i]:.1%} of variance):")
        abs_loadings = loadings.loc[pc].abs().sort_values(ascending=False)
        top_features = abs_loadings.head(3)
        for feature, loading in top_features.items():
            direction = "positively" if loadings.loc[pc, feature] > 0 else "negatively"
            print(f"  - Most {direction} influenced by {feature} ({loading:.3f})")


def test_pca_correlation_with_sensitive_attrs(pca_df, df, sensitive_attrs):
    print("\nCORRELATION OF PCA COMPONENTS WITH SENSITIVE ATTRIBUTES")
    print("=" * 60)

    results = {}

    for attr in sensitive_attrs:
        if df[attr].dtype == 'object':
            df_encoded = df[attr].astype('category').cat.codes
        else:
            df_encoded = df[attr]

        for pc in pca_df.columns:
            corr = np.corrcoef(pca_df[pc], df_encoded)[0, 1]
            print(f"Correlation between {pc} and {attr}: {corr:.3f}")
            results[(pc, attr)] = corr

    return results


def run_fairness_tests(pca_df, df, group_col):
    print(f"\nFAIRNESS TESTING BY GROUP: {group_col}")
    print("=" * 50)

    test_results = {}

    groups = df[group_col].dropna().unique()
    for pc in pca_df.columns:
        data_by_group = [pca_df.loc[df[group_col] == group, pc] for group in groups]

        # Choose test based on data distribution (Kruskal safer for non-normal)
        stat, p_value = kruskal(*data_by_group)
        test_results[pc] = p_value
        print(f"{pc}: Fairness test p = {p_value:.4f}")

    return test_results


def visualize_pca_bias(pca_df, df, group_col):
    print(f"\nVISUALIZING DISTRIBUTION OF PCA COMPONENTS BY {group_col}")
    print("=" * 60)

    for pc in pca_df.columns[:4]:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=df[group_col], y=pca_df[pc])
        plt.title(f"{pc} Distribution by {group_col}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{pc}_by_{group_col}_from_db.png")
        plt.show()


def generate_comprehensive_report_db(db_name="emergency_calls.db"):
    """
    Generate a comprehensive temporal analysis report using database.

    Parameters:
    db_name (str): Database file name
    """
    print("\n" + "=" * 80)
    print("COMPREHENSIVE TEMPORAL ANALYSIS REPORT (FROM DATABASE)")
    print("=" * 80)

    # Load data from database
    df = load_and_prepare_data_from_db(db_name)
    if df is None:
        print("Error: Could not load data from database")
        return None

    # Get summary statistics from database
    summary_stats = get_summary_stats_from_db(db_name)

    total_calls = len(df)
    date_range = f"{df['timeStamp'].min().date()} to {df['timeStamp'].max().date()}"

    print(f"\nDATASET OVERVIEW:")
    print(f"  Total emergency calls: {int(total_calls):,d}")
    print(f"  Date range: {date_range}")
    print(f"  Analysis period: {(df['timeStamp'].max() - df['timeStamp'].min()).days} days")

    # Run database-specific analyses
    analyze_missing_zip_codes_db(db_name)
    hourly_results = analyze_hourly_patterns_db(db_name)
    analyze_missing_values_db(db_name)
    detect_outliers_db(db_name)
    analyze_geospatial_db(db_name)

    # Run analyses that use loaded DataFrame
    daily_results = analyze_daily_patterns(df)
    monthly_results = analyze_monthly_seasonal_patterns(df)
    period_results = analyze_time_periods(df)

    # Create visualizations
    create_visualizations(df)
    bivariate_analysis(df)
    correlation_heatmap(df)
    plot_time_series(df)

    # Advanced analytics
    pca_df, pca_model, pca_features = apply_pca(df, n_components=0.95)
    explain_pca_components(pca_model, pca_features)

    sensitive_attributes = ['hour', 'category', 'zip']
    test_pca_correlation_with_sensitive_attrs(pca_df, df, sensitive_attributes)
    run_fairness_tests(pca_df, df, group_col='category')
    visualize_pca_bias(pca_df, df, group_col='category')

    # Summary of key findings
    print(f"\nKEY FINDINGS SUMMARY:")
    print("=" * 30)

    if hourly_results:
        peak_hour = hourly_results['hourly_counts'].idxmax()
        peak_count = hourly_results['hourly_counts'].max()
        evening_pct = hourly_results['evening_percentage']

        print(f"  Peak hour: {peak_hour}:00 with {int(peak_count):,d} calls")
        print(f"  Evening hours (17:00-19:59): {evening_pct:.1f}% of all calls")

    if daily_results:
        busiest_day = daily_results['daily_counts'].idxmax()
        weekday_avg = daily_results['weekday_count'] / 5
        weekend_avg = daily_results['weekend_count'] / 2

        print(f"  Busiest day: {busiest_day}")
        print(f"  Weekday average: {weekday_avg:,.0f} calls/day")
        print(f"  Weekend average: {weekend_avg:,.0f} calls/day")

    # Display database summary statistics
    if 'zip_stats' in summary_stats and not summary_stats['zip_stats'].empty:
        print(f"\nTOP ZIP CODES FROM DATABASE:")
        for _, row in summary_stats['zip_stats'].head(5).iterrows():
            print(f"  {row['zip_code']}: {row['total_calls']:,d} calls (peak: {row['peak_hour']}:00)")

    if 'category_stats' in summary_stats and not summary_stats['category_stats'].empty:
        print(f"\nEMERGENCY CATEGORIES FROM DATABASE:")
        for _, row in summary_stats['category_stats'].iterrows():
            print(f"  {row['category']}: {row['total_calls']:,d} calls ({row['percentage']:.1f}%)")

    return {
        'hourly': hourly_results,
        'daily': daily_results,
        'monthly': monthly_results,
        'periods': period_results,
        'database_stats': summary_stats,
        'summary_stats': {
            'total_calls': total_calls,
            'peak_hour': hourly_results.get('hourly_counts', pd.Series()).idxmax() if hourly_results else None,
            'evening_percentage': hourly_results.get('evening_percentage', 0) if hourly_results else 0,
            'busiest_day': daily_results.get('daily_counts', pd.Series()).idxmax() if daily_results else None
        }
    }


def demonstrate_sql_queries(db_name="emergency_calls.db"):
    """
    Demonstrate various SQL queries for analysis.
    """
    print("\n" + "=" * 60)
    print("DEMONSTRATING SQL QUERIES FOR ANALYSIS")
    print("=" * 60)

    conn = connect_to_database(db_name)
    if conn is None:
        return

    try:
        # Query 1: Peak hours analysis
        print("\n1. PEAK HOURS BY EMERGENCY TYPE:")
        query1 = """
        SELECT category, hour, COUNT(*) as call_count
        FROM emergency_calls 
        WHERE category IN ('EMS', 'Fire', 'Traffic')
        GROUP BY category, hour
        HAVING call_count = (
            SELECT MAX(hourly_count) 
            FROM (
                SELECT COUNT(*) as hourly_count 
                FROM emergency_calls e2 
                WHERE e2.category = emergency_calls.category 
                GROUP BY hour
            )
        )
        ORDER BY category
        """
        result1 = pd.read_sql_query(query1, conn)
        for _, row in result1.iterrows():
            print(f"   {row['category']}: Peak at {row['hour']}:00 with {row['call_count']} calls")

        # Query 2: Monthly trends
        print("\n2. MONTHLY CALL TRENDS:")
        query2 = """
        SELECT month_name, year, COUNT(*) as calls
        FROM emergency_calls
        GROUP BY year, month, month_name
        ORDER BY year, month
        LIMIT 12
        """
        result2 = pd.read_sql_query(query2, conn)
        for _, row in result2.iterrows():
            print(f"   {row['month_name']} {row['year']}: {row['calls']:,d} calls")

        # Query 3: Geographic hotspots
        print("\n3. GEOGRAPHIC HOTSPOTS (Top ZIP codes with coordinates):")
        query3 = """
        SELECT z.zip_code, z.total_calls, z.dominant_category,
               ROUND(AVG(e.lat), 4) as avg_lat, 
               ROUND(AVG(e.lng), 4) as avg_lng
        FROM zip_code_stats z
        JOIN emergency_calls e ON z.zip_code = e.zip_code
        WHERE e.lat IS NOT NULL AND e.lng IS NOT NULL
        GROUP BY z.zip_code, z.total_calls, z.dominant_category
        ORDER BY z.total_calls DESC
        LIMIT 5
        """
        result3 = pd.read_sql_query(query3, conn)
        for _, row in result3.iterrows():
            print(
                f"   {row['zip_code']}: {row['total_calls']:,d} calls ({row['dominant_category']}) at ({row['avg_lat']}, {row['avg_lng']})")

        # Query 4: Weekend vs Weekday patterns
        print("\n4. WEEKEND VS WEEKDAY PATTERNS:")
        query4 = """
        SELECT 
            CASE 
                WHEN day_of_week IN ('Saturday', 'Sunday') THEN 'Weekend'
                ELSE 'Weekday'
            END as period_type,
            category,
            COUNT(*) as calls,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(PARTITION BY category), 1) as percentage
        FROM emergency_calls
        WHERE category IN ('EMS', 'Fire', 'Traffic')
        GROUP BY period_type, category
        ORDER BY category, period_type
        """
        result4 = pd.read_sql_query(query4, conn)
        for _, row in result4.iterrows():
            print(f"   {row['category']} - {row['period_type']}: {row['calls']:,d} calls ({row['percentage']}%)")

        # Query 5: Seasonal emergency patterns
        print("\n5. SEASONAL EMERGENCY PATTERNS:")
        query5 = """
        SELECT season, category, COUNT(*) as calls
        FROM emergency_calls
        WHERE category IN ('EMS', 'Fire', 'Traffic')
        GROUP BY season, category
        ORDER BY season, calls DESC
        """
        result5 = pd.read_sql_query(query5, conn)
        current_season = None
        for _, row in result5.iterrows():
            if row['season'] != current_season:
                print(f"\n   {row['season']}:")
                current_season = row['season']
            print(f"     {row['category']}: {row['calls']:,d} calls")

        conn.close()

    except Exception as e:
        print(f"Error running SQL queries: {e}")
        conn.close()


def quick_database_check(db_name='emergency_calls.db'):
    """
    Quick function to verify database is working and show basic stats.
    """
    print("QUICK DATABASE CHECK")
    print("=" * 25)

    conn = connect_to_database(db_name)
    if conn is None:
        print("Database connection failed")
        return False

    try:
        # Check if tables exist
        tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
        tables = pd.read_sql_query(tables_query, conn)

        print(f"Database connected successfully")
        print(f"Tables found: {', '.join(tables['name'].tolist())}")

        # Quick stats
        stats_query = """
        SELECT 
            (SELECT COUNT(*) FROM emergency_calls) as total_calls,
            (SELECT COUNT(DISTINCT zip_code) FROM emergency_calls WHERE zip_code IS NOT NULL) as unique_zips,
            (SELECT COUNT(DISTINCT category) FROM emergency_calls) as categories,
            (SELECT MIN(DATE(timestamp)) FROM emergency_calls) as earliest_date,
            (SELECT MAX(DATE(timestamp)) FROM emergency_calls) as latest_date
        """
        stats = pd.read_sql_query(stats_query, conn)

        print(f"Total calls: {stats['total_calls'].iloc[0]:,d}")
        print(f"Unique ZIP codes: {stats['unique_zips'].iloc[0]}")
        print(f"Emergency categories: {stats['categories'].iloc[0]}")
        print(f"Date range: {stats['earliest_date'].iloc[0]} to {stats['latest_date'].iloc[0]}")

        conn.close()
        return True

    except Exception as e:
        print(f"Error checking database: {e}")
        conn.close()
        return False


if __name__ == "__main__":
    print("911 EMERGENCY CALLS ANALYSIS - DATABASE INTEGRATED VERSION")
    print("=" * 65)

    # Check if database exists
    db_name = 'emergency_calls.db'

    if not os.path.exists(db_name):
        exit(1)

    # Quick database check
    if not quick_database_check(db_name):
        exit(1)

    print(f"Starting comprehensive analysis using database...")

    try:
        # Run comprehensive analysis
        results = generate_comprehensive_report_db(db_name)

        # Demonstrate SQL capabilities
        demonstrate_sql_queries(db_name)

        print(f"\nANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"All data sourced from database: {db_name}")

        if results:
            print(f"\nQUICK SUMMARY:")
            summary = results['summary_stats']
            print(f"Total emergency calls analyzed: {summary['total_calls']:,d}")
            if summary['peak_hour']:
                print(f"Peak activity hour: {summary['peak_hour']}:00")
            print(f"Evening concentration: {summary['evening_percentage']:.1f}%")
            if summary['busiest_day']:
                print(f"Busiest day of week: {summary['busiest_day']}")

    except FileNotFoundError:
        print("Error: Required files not found")
        print("Make sure both the database and any required Python packages are available")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback

        traceback.print_exc()