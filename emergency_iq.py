import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from scipy.stats import kruskal, f_oneway
import sqlite3
import warnings
import os

warnings.filterwarnings('ignore')


# =============================================================================
# DATABASE SETUP FUNCTIONS
# =============================================================================

def inspect_data_structure(csv_file='911.csv'):
    """Inspect the structure of the CSV data."""
    print("INSPECTING DATA STRUCTURE")
    print("=" * 40)

    df = pd.read_csv(csv_file)

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nColumn data types:")
    print(df.dtypes)
    print(f"\nFirst few rows:")
    print(df.head())

    print(f"\nMissing values:")
    print(df.isnull().sum())

    print(f"\nSample values for each column:")
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_vals = df[col].dropna().unique()[:3]
            print(f"  {col}: {list(unique_vals)}")
        else:
            print(f"  {col}: {df[col].describe()}")

    return df


def create_database_schema():
    """Create database schema based on actual data structure."""
    schema_sql = """
    -- Drop tables if they exist
    DROP TABLE IF EXISTS emergency_calls;
    DROP TABLE IF EXISTS zip_code_stats;
    DROP TABLE IF EXISTS temporal_patterns;
    DROP TABLE IF EXISTS emergency_types;

    -- Main emergency calls table
    CREATE TABLE emergency_calls (
        call_id INTEGER PRIMARY KEY AUTOINCREMENT,
        lat REAL,
        lng REAL,
        description TEXT,
        zip_code TEXT,
        title TEXT,
        timestamp DATETIME,
        township TEXT,
        address TEXT,
        emergency_code INTEGER,
        -- Derived columns
        hour INTEGER,
        day_of_week TEXT,
        day_of_week_num INTEGER,
        month INTEGER,
        month_name TEXT,
        year INTEGER,
        season TEXT,
        category TEXT,
        time_period TEXT
    );

    -- ZIP code statistics table
    CREATE TABLE zip_code_stats (
        zip_code TEXT PRIMARY KEY,
        total_calls INTEGER,
        avg_calls_per_day REAL,
        peak_hour INTEGER,
        dominant_category TEXT,
        first_call_date DATE,
        last_call_date DATE
    );

    -- Temporal patterns table
    CREATE TABLE temporal_patterns (
        pattern_id INTEGER PRIMARY KEY AUTOINCREMENT,
        pattern_type TEXT,  -- 'hourly', 'daily', 'monthly', 'seasonal'
        pattern_value TEXT, -- actual hour/day/month/season
        call_count INTEGER,
        percentage REAL
    );

    -- Emergency types table
    CREATE TABLE emergency_types (
        category TEXT PRIMARY KEY,
        total_calls INTEGER,
        percentage REAL,
        avg_calls_per_day REAL,
        peak_hour INTEGER
    );

    -- Create indexes for performance
    CREATE INDEX idx_timestamp ON emergency_calls(timestamp);
    CREATE INDEX idx_zip_code ON emergency_calls(zip_code);
    CREATE INDEX idx_category ON emergency_calls(category);
    CREATE INDEX idx_hour ON emergency_calls(hour);
    CREATE INDEX idx_lat_lng ON emergency_calls(lat, lng);
    """
    return schema_sql


def prepare_data_for_database(df):
    """Prepare and clean data for database insertion."""
    print("\nPREPARING DATA FOR DATABASE")
    print("=" * 35)

    df_clean = df.copy()

    print("Converting timestamp...")
    df_clean['timeStamp'] = pd.to_datetime(df_clean['timeStamp'])

    print("Extracting temporal features...")
    df_clean['hour'] = df_clean['timeStamp'].dt.hour
    df_clean['day_of_week'] = df_clean['timeStamp'].dt.day_name()
    df_clean['day_of_week_num'] = df_clean['timeStamp'].dt.dayofweek
    df_clean['month'] = df_clean['timeStamp'].dt.month
    df_clean['month_name'] = df_clean['timeStamp'].dt.month_name()
    df_clean['year'] = df_clean['timeStamp'].dt.year

    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'

    df_clean['season'] = df_clean['month'].apply(get_season)

    def get_time_period(hour):
        if 6 <= hour < 12:
            return 'Morning (6-12)'
        elif 12 <= hour < 17:
            return 'Afternoon (12-17)'
        elif 17 <= hour < 21:
            return 'Evening (17-21)'
        else:
            return 'Night (21-6)'

    df_clean['time_period'] = df_clean['hour'].apply(get_time_period)

    def categorize_title(title):
        if pd.isna(title):
            return 'Other'
        title_str = str(title).upper()
        if 'EMS' in title_str or 'MEDICAL' in title_str or 'AMBULANCE' in title_str:
            return 'EMS'
        elif 'FIRE' in title_str:
            return 'Fire'
        elif 'TRAFFIC' in title_str or 'VEHICLE' in title_str or 'ACCIDENT' in title_str:
            return 'Traffic'
        else:
            return 'Other'

    df_clean['category'] = df_clean['title'].apply(categorize_title)

    print("Cleaning ZIP codes...")
    df_clean['zip'] = df_clean['zip'].astype(str)
    df_clean['zip'] = df_clean['zip'].replace('nan', None)

    print("Handling missing values...")
    missing_before = df_clean.isnull().sum().sum()
    df_clean['twp'] = df_clean['twp'].fillna('Unknown')
    df_clean['addr'] = df_clean['addr'].fillna('Unknown Address')
    missing_after = df_clean.isnull().sum().sum()
    print(f"Reduced missing values from {missing_before} to {missing_after}")

    df_clean = df_clean.rename(columns={
        'desc': 'description',
        'zip': 'zip_code',
        'timeStamp': 'timestamp',
        'twp': 'township',
        'addr': 'address',
        'e': 'emergency_code'
    })

    print(f"Data prepared: {len(df_clean):,} records ready for database")
    return df_clean


def create_database_connection(db_name='emergency_calls.db'):
    """Create database connection and execute schema."""
    print(f"\nCREATING DATABASE: {db_name}")
    print("=" * 30)

    if os.path.exists(db_name):
        os.remove(db_name)
        print(f"Removed existing database: {db_name}")

    conn = sqlite3.connect(db_name)
    schema = create_database_schema()
    conn.executescript(schema)
    print("Database schema created successfully")
    return conn


def create_summary_tables(conn, df):
    """Create aggregated summary tables."""
    print(f"\nCREATING SUMMARY TABLES")
    print("=" * 25)

    cursor = conn.cursor()

    print("Creating ZIP code statistics...")
    zip_stats = df.groupby('zip_code').agg({
        'timestamp': ['count', 'min', 'max'],
        'hour': lambda x: x.mode().iloc[0] if len(x) > 0 else 0,
        'category': lambda x: x.mode().iloc[0] if len(x) > 0 else 'Unknown'
    }).round(2)

    zip_stats.columns = ['total_calls', 'first_call_date', 'last_call_date', 'peak_hour', 'dominant_category']
    zip_stats['avg_calls_per_day'] = zip_stats['total_calls'] / ((pd.to_datetime(
        zip_stats['last_call_date']) - pd.to_datetime(zip_stats['first_call_date'])).dt.days + 1)
    zip_stats = zip_stats.reset_index()
    zip_stats = zip_stats[zip_stats['zip_code'].notna()]
    zip_stats.to_sql('zip_code_stats', conn, if_exists='replace', index=False)
    print(f"ZIP code stats: {len(zip_stats)} ZIP codes")

    print("Creating temporal patterns...")

    # Hourly patterns
    hourly_counts = df['hour'].value_counts().reset_index()
    hourly_counts['pattern_type'] = 'hourly'
    hourly_counts['pattern_value'] = hourly_counts['hour'].astype(str)
    hourly_counts['percentage'] = (hourly_counts['count'] / len(df)) * 100
    hourly_counts = hourly_counts[['pattern_type', 'pattern_value', 'count', 'percentage']]
    hourly_counts.columns = ['pattern_type', 'pattern_value', 'call_count', 'percentage']

    # Daily patterns
    daily_counts = df['day_of_week'].value_counts().reset_index()
    daily_counts['pattern_type'] = 'daily'
    daily_counts['pattern_value'] = daily_counts['day_of_week']
    daily_counts['percentage'] = (daily_counts['count'] / len(df)) * 100
    daily_counts = daily_counts[['pattern_type', 'pattern_value', 'count', 'percentage']]
    daily_counts.columns = ['pattern_type', 'pattern_value', 'call_count', 'percentage']

    # Monthly patterns
    monthly_counts = df['month_name'].value_counts().reset_index()
    monthly_counts['pattern_type'] = 'monthly'
    monthly_counts['pattern_value'] = monthly_counts['month_name']
    monthly_counts['percentage'] = (monthly_counts['count'] / len(df)) * 100
    monthly_counts = monthly_counts[['pattern_type', 'pattern_value', 'count', 'percentage']]
    monthly_counts.columns = ['pattern_type', 'pattern_value', 'call_count', 'percentage']

    # Seasonal patterns
    seasonal_counts = df['season'].value_counts().reset_index()
    seasonal_counts['pattern_type'] = 'seasonal'
    seasonal_counts['pattern_value'] = seasonal_counts['season']
    seasonal_counts['percentage'] = (seasonal_counts['count'] / len(df)) * 100
    seasonal_counts = seasonal_counts[['pattern_type', 'pattern_value', 'count', 'percentage']]
    seasonal_counts.columns = ['pattern_type', 'pattern_value', 'call_count', 'percentage']

    all_patterns = pd.concat([hourly_counts, daily_counts, monthly_counts, seasonal_counts], ignore_index=True)
    all_patterns.to_sql('temporal_patterns', conn, if_exists='replace', index=False)
    print(f"Temporal patterns: {len(all_patterns)} patterns")

    print("Creating emergency type statistics...")
    category_stats = df.groupby('category').agg({
        'timestamp': 'count',
        'hour': lambda x: x.mode().iloc[0] if len(x) > 0 else 0
    }).reset_index()

    category_stats.columns = ['category', 'total_calls', 'peak_hour']
    category_stats['percentage'] = (category_stats['total_calls'] / len(df)) * 100

    date_range_days = (df['timestamp'].max() - df['timestamp'].min()).days + 1
    category_stats['avg_calls_per_day'] = category_stats['total_calls'] / date_range_days

    category_stats.to_sql('emergency_types', conn, if_exists='replace', index=False)
    print(f"Emergency types: {len(category_stats)} categories")

    conn.commit()
    print(f"All summary tables created successfully")


def setup_database_from_csv(csv_file='911.csv', db_name='emergency_calls.db'):
    """Main function to load data into database."""
    df = inspect_data_structure(csv_file)
    df_clean = prepare_data_for_database(df)
    conn = create_database_connection(db_name)

    print(f"\nLOADING DATA TO DATABASE")
    print("=" * 30)

    db_columns = [
        'lat', 'lng', 'description', 'zip_code', 'title', 'timestamp',
        'township', 'address', 'emergency_code', 'hour', 'day_of_week',
        'day_of_week_num', 'month', 'month_name', 'year', 'season',
        'category', 'time_period'
    ]

    df_for_db = df_clean[db_columns].copy()
    df_for_db.to_sql('emergency_calls', conn, if_exists='append', index=False)
    print(f"Loaded {len(df_for_db):,} records to emergency_calls table")

    create_summary_tables(conn, df_clean)
    return conn, df_clean


# =============================================================================
# DATABASE CONNECTION AND LOADING FUNCTIONS
# =============================================================================

def connect_to_database(db_name='emergency_calls.db'):
    """Create connection to the SQLite database."""
    try:
        conn = sqlite3.connect(db_name)
        print(f"Connected to database: {db_name}")
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None


def load_and_prepare_data_from_db(db_name="emergency_calls.db"):
    """Load the 911 calls data from database with temporal features already prepared."""
    print("Loading data from database...")

    conn = connect_to_database(db_name)
    if conn is None:
        return None

    try:
        query = """
        SELECT call_id, lat, lng, description, zip_code, title, timestamp,
               township, address, emergency_code, hour, day_of_week,
               day_of_week_num, month, month_name, year, season, 
               category, time_period
        FROM emergency_calls
        ORDER BY timestamp
        """

        df = pd.read_sql_query(query, conn)
        df['timeStamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timeStamp'].dt.date
        df['week_of_year'] = df['timeStamp'].dt.isocalendar().week

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
    """Get pre-computed summary statistics from database tables."""
    conn = connect_to_database(db_name)
    if conn is None:
        return {}

    try:
        temporal_query = """
        SELECT pattern_type, pattern_value, call_count, percentage
        FROM temporal_patterns
        ORDER BY pattern_type, call_count DESC
        """
        temporal_df = pd.read_sql_query(temporal_query, conn)

        zip_query = """
        SELECT zip_code, total_calls, avg_calls_per_day, peak_hour, dominant_category
        FROM zip_code_stats
        ORDER BY total_calls DESC
        LIMIT 10
        """
        zip_df = pd.read_sql_query(zip_query, conn)

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


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_missing_zip_codes_db(db_name='emergency_calls.db'):
    """Analyze missing ZIP codes using database queries."""
    print("Analyzing missing ZIP codes from database...")

    conn = connect_to_database(db_name)
    if conn is None:
        return None

    try:
        total_query = "SELECT COUNT(*) FROM emergency_calls"
        total_records = pd.read_sql_query(total_query, conn).iloc[0, 0]

        missing_query = """
        SELECT COUNT(*) FROM emergency_calls 
        WHERE zip_code IS NULL OR zip_code = '' OR zip_code = 'None'
        """
        total_missing = pd.read_sql_query(missing_query, conn).iloc[0, 0]

        valid_zip_codes = total_records - total_missing
        missing_percentage = (total_missing / total_records) * 100
        valid_percentage = (valid_zip_codes / total_records) * 100

        sample_query = """
        SELECT township, address, title 
        FROM emergency_calls 
        WHERE zip_code IS NULL OR zip_code = '' OR zip_code = 'None'
        LIMIT 10
        """
        missing_sample = pd.read_sql_query(sample_query, conn)

        coords_query = """
        SELECT COUNT(*) FROM emergency_calls 
        WHERE lat IS NOT NULL AND lng IS NOT NULL
        """
        coords_available = pd.read_sql_query(coords_query, conn).iloc[0, 0]

        missing_with_coords_query = """
        SELECT COUNT(*) FROM emergency_calls 
        WHERE (zip_code IS NULL OR zip_code = '' OR zip_code = 'None')
        AND lat IS NOT NULL AND lng IS NOT NULL
        """
        missing_with_coords = pd.read_sql_query(missing_with_coords_query, conn).iloc[0, 0]

        conn.close()

        results = {
            'total_records': total_records,
            'total_missing': total_missing,
            'valid_zip_codes': valid_zip_codes,
            'missing_percentage': missing_percentage,
            'valid_percentage': valid_percentage,
            'coords_available': coords_available,
            'missing_with_coords': missing_with_coords
        }

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
    """Analyze call volume patterns by hour using database queries."""
    print("\nANALYZING HOURLY PATTERNS (FROM DATABASE)")
    print("=" * 45)

    conn = connect_to_database(db_name)
    if conn is None:
        return {}

    try:
        hourly_query = """
        SELECT pattern_value, call_count, percentage
        FROM temporal_patterns 
        WHERE pattern_type = 'hourly'
        ORDER BY CAST(pattern_value AS INTEGER)
        """
        hourly_data = pd.read_sql_query(hourly_query, conn)

        hourly_counts = pd.Series(
            hourly_data['call_count'].values,
            index=hourly_data['pattern_value'].astype(int)
        )

        peak_hours = hourly_counts.nlargest(5)
        quiet_hours = hourly_counts.nsmallest(5)

        evening_hours = [17, 18, 19]
        evening_query = """
        SELECT SUM(call_count) FROM temporal_patterns
        WHERE pattern_type = 'hourly' AND pattern_value IN ('17', '18', '19')
        """
        evening_count = pd.read_sql_query(evening_query, conn).iloc[0, 0]

        total_query = "SELECT COUNT(*) FROM emergency_calls"
        total_calls = pd.read_sql_query(total_query, conn).iloc[0, 0]
        evening_percentage = (evening_count / total_calls) * 100

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


def analyze_daily_patterns(df):
    """Analyze call volume patterns by day of week."""
    print("\nANALYZING DAILY PATTERNS")
    print("=" * 40)

    daily_counts = df['day_of_week'].value_counts()
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
    """Analyze call volume patterns by month and season."""
    print("\nANALYZING MONTHLY AND SEASONAL PATTERNS")
    print("=" * 50)

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
    """Analyze call volume patterns by time periods."""
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
    """Create temporal analysis visualizations."""
    print("\nCREATING VISUALIZATIONS")
    print("=" * 30)

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

    # 6. Category distribution by hour
    plt.subplot(2, 3, 6)
    category_hour = pd.crosstab(df['category'], df['hour'])
    category_hour_pct = category_hour.div(category_hour.sum(axis=1), axis=0) * 100

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

def apply_pca(df, n_components=0.95):
    """Apply PCA for dimensionality reduction."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    clean_cols = [col for col in numeric_cols if df[col].nunique() > 1 and df[col].notna().all()]
    X = df[clean_cols]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(X_pca, columns=[f"PC{i + 1}" for i in range(X_pca.shape[1])])

    print("\nAPPLYING PCA FOR DIMENSIONALITY REDUCTION")

    print(f"Numeric features used for PCA: {clean_cols}")
    print(f"Original feature count: {len(clean_cols)}")
    print(f"Reduced to {pca.n_components_} components (explained variance ≥ {n_components})")

    return pca_df, pca, clean_cols


def perform_dbscan_clustering(df, pca_df, use_full_dataset=True, target_clusters=8):
    """Perform DBSCAN clustering on PCA components - optimized for target number of clusters."""
    print(f"\nPERFORMING DBSCAN CLUSTERING - TARGET: {target_clusters} CLUSTERS")

    print(f"Dataset size: {len(df):,} records")
    # Use all data points
    pca_full = pca_df.copy()
    df_full = df.copy()

    # Select first 4 PCA components and convert to memory-efficient format
    feature_cols = ['PC1', 'PC2', 'PC3', 'PC4']
    X = pca_full[feature_cols].values.astype(np.float32)  # Use float32 to save memory

    print(f"Using features: {feature_cols}")
    print(f"Data shape: {X.shape}")
    print(f"Memory usage: {X.nbytes / 1024 ** 2:.1f} MB")

    # Standardize features
    scaler = StandardScaler()
    print("Standardizing features...")
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    # Free original data to save memory
    del X

    # Parameter tuning to achieve target number of clusters
    print(f"PARAMETER TUNING FOR {target_clusters} CLUSTERS")
    print("=" * 55)

    # Test different eps values to find optimal number of clusters
    min_samples = 10  # Increased for fewer, more robust clusters
    eps_candidates = [0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]

    best_eps = None
    best_n_clusters = 0
    best_score = float('inf')

    print("Testing eps values to find optimal clustering...")

    for eps in eps_candidates:
        print(f"Testing eps={eps:.1f}...", end="")

        # Test on a sample for speed
        sample_size = min(50000, len(X_scaled))
        sample_indices = np.random.choice(len(X_scaled), sample_size, replace=False)
        X_sample = X_scaled[sample_indices]

        dbscan_test = DBSCAN(eps=eps, min_samples=min_samples, algorithm='ball_tree', n_jobs=-1)
        test_clusters = dbscan_test.fit_predict(X_sample)

        n_clusters = len(set(test_clusters)) - (1 if -1 in test_clusters else 0)
        n_noise = list(test_clusters).count(-1)
        noise_ratio = n_noise / len(test_clusters)

        # Score based on how close to target and noise ratio
        cluster_diff = abs(n_clusters - target_clusters)
        score = cluster_diff + (noise_ratio * 10)  # Penalize high noise

        print(f" {n_clusters} clusters, {noise_ratio:.1%} noise")

        if score < best_score:
            best_score = score
            best_eps = eps
            best_n_clusters = n_clusters

    print(f"\nOPTIMAL PARAMETERS FOUND:")
    print(f"   Best eps: {best_eps}")
    print(f"   Expected clusters: {best_n_clusters}")
    print(f"   Min samples: {min_samples}")

    # Apply DBSCAN with optimized parameters on full dataset
    print(f"\nAPPLYING OPTIMIZED DBSCAN TO FULL DATASET")

    dbscan = DBSCAN(
        eps=best_eps,
        min_samples=min_samples,
        algorithm='ball_tree',
        n_jobs=-1
    )

    # Perform clustering on full dataset
    clusters = dbscan.fit_predict(X_scaled)

    # Free scaled data to save memory
    del X_scaled

    # Add cluster labels to dataframes
    pca_full['cluster'] = clusters
    df_full['cluster'] = clusters

    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)

    print(f"CLUSTERING COMPLETED!")
    print(f"   Final clusters: {n_clusters}")
    print(f"   Noise points: {n_noise:,} ({n_noise / len(clusters) * 100:.1f}%)")
    print(f"   Target achieved: {'YES' if abs(n_clusters - target_clusters) <= 2 else 'CLOSE'}")

    # Visualization using ALL data points
    print(f"Creating visualization using ALL {len(pca_full):,} data points...")

    plt.figure(figsize=(15, 6))

    # Plot 1: DBSCAN results
    plt.subplot(1, 2, 1)
    scatter1 = plt.scatter(pca_full['PC3'], pca_full['PC4'], c=pca_full['cluster'],
                           cmap='tab10', s=1, alpha=0.5)
    plt.xlabel('PC3')
    plt.ylabel('PC4')
    plt.title(f'DBSCAN Clusters\n({len(df_full):,} points, {n_clusters} clusters)')
    plt.colorbar(scatter1, label='Cluster label')
    plt.grid(True, alpha=0.3)

    plt.savefig('dbscan_results.png', dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

    # Choose best clustering result
    final_clusters = clusters
    cluster_method = "DBSCAN"

    # Comprehensive cluster analysis on full dataset
    print(f"\nFINAL CLUSTER ANALYSIS ({cluster_method})")
    print("=" * 50)

    cluster_counts = df_full['cluster'].value_counts().sort_index()
    print(f"Final clustering method: {cluster_method}")
    print(f"Number of clusters: {n_clusters}")
    print(f"Cluster sizes:")
    for cluster_id, count in cluster_counts.items():
        percentage = (count / len(df_full)) * 100
        print(f"  Cluster {cluster_id:2d}: {count:8,d} calls ({percentage:5.1f}%)")

    # Detailed cluster statistics
    print(f"\nDetailed cluster characteristics:")
    cluster_stats = []

    for cluster_id in sorted(df_full['cluster'].unique()):
        if cluster_id == -1:  # Skip noise if DBSCAN
            continue

        cluster_data = df_full[df_full['cluster'] == cluster_id]

        # Calculate statistics
        stats = {
            'cluster': cluster_id,
            'size': len(cluster_data),
            'percentage': len(cluster_data) / len(df_full) * 100,
            'avg_hour': cluster_data['hour'].mean(),
            'std_hour': cluster_data['hour'].std(),
            'avg_day': cluster_data['day_of_week_num'].mean(),
            'top_category': cluster_data['category'].mode().iloc[0] if len(cluster_data) > 0 else 'Unknown',
            'category_purity': cluster_data['category'].value_counts().iloc[0] / len(cluster_data) * 100 if len(
                cluster_data) > 0 else 0
        }

        # Add location stats if available
        if 'lat' in cluster_data.columns and cluster_data['lat'].notna().sum() > 0:
            stats['avg_lat'] = cluster_data['lat'].mean()
            stats['avg_lng'] = cluster_data['lng'].mean()
            stats['lat_std'] = cluster_data['lat'].std()
            stats['lng_std'] = cluster_data['lng'].std()

        cluster_stats.append(stats)

    # Display cluster characteristics
    for stats in cluster_stats:
        print(f"\n--- Cluster {stats['cluster']} ---")
        print(f"Size: {stats['size']:,} calls ({stats['percentage']:.1f}%)")
        print(f"Time pattern: Hour {stats['avg_hour']:.1f}±{stats['std_hour']:.1f}, Day {stats['avg_day']:.1f}")
        print(f"Dominant category: {stats['top_category']} ({stats['category_purity']:.1f}% purity)")

        if 'avg_lat' in stats:
            print(
                f"Location: ({stats['avg_lat']:.4f}±{stats['lat_std']:.4f}, {stats['avg_lng']:.4f}±{stats['lng_std']:.4f})")

    # Create enhanced visualizations
    print(f"\nGenerating cluster analysis visualizations...")

    # Hourly activity heatmap
    plt.figure(figsize=(15, max(8, n_clusters * 0.6)))
    hourly_cluster = pd.crosstab(df_full['cluster'], df_full['hour'])
    sns.heatmap(hourly_cluster, annot=False, cmap='YlOrRd', cbar_kws={'label': 'Number of Calls'})
    plt.title(f'Hourly Activity Pattern by Cluster ({cluster_method})\n{len(df_full):,} calls, {n_clusters} clusters')
    plt.xlabel('Hour of Day')
    plt.ylabel('Cluster ID')
    plt.savefig('temporal_cluster_results.png', dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

    # Category distribution by cluster
    plt.figure(figsize=(12, 8))
    category_cluster = pd.crosstab(df_full['category'], df_full['cluster'], normalize='columns') * 100

    sns.heatmap(category_cluster, annot=True, fmt='.1f', cmap='Blues',
                cbar_kws={'label': 'Percentage within Cluster'})
    plt.title(f'Emergency Category Distribution by Cluster ({cluster_method})')
    plt.xlabel('Cluster ID')
    plt.ylabel('Emergency Category')
    plt.savefig('emergency_category.png', dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

    return final_clusters, df_full, cluster_stats


def perform_dbscan_clustering_fair(df, pca_df, use_full_dataset=True, target_clusters=8):
    """Perform DBSCAN clustering on PC4 and PC5 components (temporal features) - fairness-improved version."""
    print(f"\nFAIRNESS-IMPROVED DBSCAN CLUSTERING - USING PC4 & PC5 (TEMPORAL FEATURES)")
    print(f"TARGET: {target_clusters} CLUSTERS")

    print(f"Dataset size: {len(df):,} records")
    # Use all data points
    pca_full = pca_df.copy()
    df_full = df.copy()

    # Select PC4 and PC5 (temporal-focused components) instead of PC3 and PC4
    feature_cols = ['PC4', 'PC5']  # These are primarily temporal features (hour, day patterns)
    X = pca_full[feature_cols].values.astype(np.float32)  # Use float32 to save memory

    print(f"Using FAIRNESS-IMPROVED features: {feature_cols}")
    print(f"Note: PC4 & PC5 focus on temporal patterns (hour, day) rather than geographic location")
    print(f"Data shape: {X.shape}")
    print(f"Memory usage: {X.nbytes / 1024 ** 2:.1f} MB")

    # Standardize features
    scaler = StandardScaler()
    print("Standardizing features...")
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    # Free original data to save memory
    del X

    # Parameter tuning to achieve target number of clusters
    print(f"PARAMETER TUNING FOR {target_clusters} CLUSTERS (TEMPORAL-FOCUSED)")
    print("=" * 65)

    # Test different eps values to find optimal number of clusters
    min_samples = 10  # Increased for fewer, more robust clusters
    eps_candidates = [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

    best_eps = None
    best_n_clusters = 0
    best_score = float('inf')

    print("Testing eps values for temporal clustering...")

    for eps in eps_candidates:
        print(f"Testing eps={eps:.1f}...", end="")

        # Test on a sample for speed
        sample_size = min(50000, len(X_scaled))
        sample_indices = np.random.choice(len(X_scaled), sample_size, replace=False)
        X_sample = X_scaled[sample_indices]

        dbscan_test = DBSCAN(eps=eps, min_samples=min_samples, algorithm='ball_tree', n_jobs=-1)
        test_clusters = dbscan_test.fit_predict(X_sample)

        n_clusters = len(set(test_clusters)) - (1 if -1 in test_clusters else 0)
        n_noise = list(test_clusters).count(-1)
        noise_ratio = n_noise / len(test_clusters)

        # Score based on how close to target and noise ratio
        cluster_diff = abs(n_clusters - target_clusters)
        score = cluster_diff + (noise_ratio * 10)  # Penalize high noise

        print(f" {n_clusters} clusters, {noise_ratio:.1%} noise")

        if score < best_score:
            best_score = score
            best_eps = eps
            best_n_clusters = n_clusters

    print(f"\nOPTIMAL TEMPORAL CLUSTERING PARAMETERS:")
    print(f"   Best eps: {best_eps}")
    print(f"   Expected clusters: {best_n_clusters}")
    print(f"   Min samples: {min_samples}")

    # Apply DBSCAN with optimized parameters on full dataset
    print(f"\nAPPLYING FAIRNESS-IMPROVED DBSCAN TO FULL DATASET")

    dbscan = DBSCAN(
        eps=best_eps,
        min_samples=min_samples,
        algorithm='ball_tree',
        n_jobs=-1
    )

    # Perform clustering on full dataset
    clusters = dbscan.fit_predict(X_scaled)

    # Free scaled data to save memory
    del X_scaled

    # Add cluster labels to dataframes
    pca_full['cluster'] = clusters
    df_full['cluster'] = clusters

    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)

    print(f"FAIRNESS-IMPROVED CLUSTERING COMPLETED!")
    print(f"   Final clusters: {n_clusters}")
    print(f"   Noise points: {n_noise:,} ({n_noise / len(clusters) * 100:.1f}%)")
    print(f"   Target achieved: {'YES' if abs(n_clusters - target_clusters) <= 2 else 'CLOSE'}")
    print(f"   Fairness benefit: Reduced geographic bias by using temporal features")

    # Visualization using ALL data points - NOW WITH PC4 and PC5
    print(f"Creating fairness-improved visualization using ALL {len(pca_full):,} data points...")

    plt.figure(figsize=(15, 6))

    # Plot 1: DBSCAN results with PC4 and PC5
    plt.subplot(1, 2, 1)
    scatter1 = plt.scatter(pca_full['PC4'], pca_full['PC5'], c=pca_full['cluster'],
                           cmap='tab10', s=1, alpha=0.5)
    plt.xlabel('PC4 (Temporal Features: Hour Patterns)')
    plt.ylabel('PC5 (Temporal Features: Day Patterns)')
    plt.title(
        f'Fairness-Improved DBSCAN Clusters (PC4 & PC5)\n({len(df_full):,} points, {n_clusters} clusters)\nTemporal-focused, reduced geographic bias')
    plt.colorbar(scatter1, label='Cluster label')
    plt.grid(True, alpha=0.3)

    # Plot 2: Comparison showing the fairness improvement
    plt.subplot(1, 2, 2)

    # Create a summary of fairness improvements
    cluster_counts = df_full['cluster'].value_counts().sort_index()
    colors = plt.cm.tab10(np.linspace(0, 1, len(cluster_counts)))

    bars = plt.bar(range(len(cluster_counts)), cluster_counts.values, color=colors, alpha=0.7)
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Calls')
    plt.title(f'Fairness-Improved Cluster Sizes\n(Based on Temporal Patterns)\nReduced Geographic Bias')
    plt.xticks(range(len(cluster_counts)), cluster_counts.index)

    # Add value labels on bars
    for bar, count in zip(bars, cluster_counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + count * 0.01,
                 f'{count:,}', ha='center', va='bottom', fontsize=8)

    plt.grid(True, alpha=0.3)

    plt.savefig('fairness_improved_dbscan_results.png', dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

    # Choose best clustering result
    final_clusters = clusters
    cluster_method = "Fairness-Improved DBSCAN (PC4+PC5)"

    # Comprehensive cluster analysis on full dataset
    print(f"\nFINAL FAIRNESS-IMPROVED CLUSTER ANALYSIS")
    print("=" * 55)

    cluster_counts = df_full['cluster'].value_counts().sort_index()
    print(f"Final clustering method: {cluster_method}")
    print(f"Number of clusters: {n_clusters}")
    print(f"Fairness improvement: Using temporal features instead of geographic")
    print(f"Cluster sizes:")
    for cluster_id, count in cluster_counts.items():
        percentage = (count / len(df_full)) * 100
        print(f"  Cluster {cluster_id:2d}: {count:8,d} calls ({percentage:5.1f}%)")

    # Detailed cluster statistics
    print(f"\nDetailed temporal cluster characteristics:")
    cluster_stats = []

    for cluster_id in sorted(df_full['cluster'].unique()):
        if cluster_id == -1:  # Skip noise if DBSCAN
            continue

        cluster_data = df_full[df_full['cluster'] == cluster_id]

        # Calculate statistics
        stats = {
            'cluster': cluster_id,
            'size': len(cluster_data),
            'percentage': len(cluster_data) / len(df_full) * 100,
            'avg_hour': cluster_data['hour'].mean(),
            'std_hour': cluster_data['hour'].std(),
            'avg_day': cluster_data['day_of_week_num'].mean(),
            'top_category': cluster_data['category'].mode().iloc[0] if len(cluster_data) > 0 else 'Unknown',
            'category_purity': cluster_data['category'].value_counts().iloc[0] / len(cluster_data) * 100 if len(
                cluster_data) > 0 else 0
        }

        # Add geographic diversity stats (to show fairness improvement)
        if 'lat' in cluster_data.columns and cluster_data['lat'].notna().sum() > 0:
            stats['geographic_spread_lat'] = cluster_data['lat'].std()
            stats['geographic_spread_lng'] = cluster_data['lng'].std()
            stats['zip_diversity'] = cluster_data['zip'].nunique() if 'zip' in cluster_data.columns else 0

        cluster_stats.append(stats)

    # Display cluster characteristics with fairness focus
    for stats in cluster_stats:
        print(f"\n--- Temporal Cluster {stats['cluster']} (Fairness-Improved) ---")
        print(f"Size: {stats['size']:,} calls ({stats['percentage']:.1f}%)")
        print(f"Temporal pattern: Hour {stats['avg_hour']:.1f}±{stats['std_hour']:.1f}, Day {stats['avg_day']:.1f}")
        print(f"Dominant category: {stats['top_category']} ({stats['category_purity']:.1f}% purity)")

        if 'geographic_spread_lat' in stats:
            print(
                f"Geographic diversity: Lat spread ±{stats['geographic_spread_lat']:.4f}, Lng spread ±{stats['geographic_spread_lng']:.4f}")
        if 'zip_diversity' in stats:
            print(f"ZIP code diversity: {stats['zip_diversity']} different ZIP codes")

    # Create enhanced visualizations with fairness focus
    print(f"\nGenerating fairness-improved cluster analysis visualizations...")

    # Hourly activity heatmap - NOW SHOWING TEMPORAL FOCUS
    plt.figure(figsize=(15, max(8, n_clusters * 0.6)))
    hourly_cluster = pd.crosstab(df_full['cluster'], df_full['hour'])
    sns.heatmap(hourly_cluster, annot=False, cmap='YlOrRd', cbar_kws={'label': 'Number of Calls'})
    plt.title(
        f'Fairness-Improved: Hourly Activity Pattern by Temporal Cluster\n{len(df_full):,} calls, {n_clusters} clusters (Based on PC4 & PC5 - Temporal Features)')
    plt.xlabel('Hour of Day')
    plt.ylabel('Temporal Cluster ID')
    plt.savefig('fairness_improved_temporal_cluster_results.png', dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

    # Category distribution by cluster
    plt.figure(figsize=(12, 8))
    category_cluster = pd.crosstab(df_full['category'], df_full['cluster'], normalize='columns') * 100

    sns.heatmap(category_cluster, annot=True, fmt='.1f', cmap='Blues',
                cbar_kws={'label': 'Percentage within Cluster'})
    plt.title(
        f'Fairness-Improved: Emergency Category Distribution by Temporal Cluster\n(PC4 & PC5 Focus - Reduced Geographic Bias)')
    plt.xlabel('Temporal Cluster ID')
    plt.ylabel('Emergency Category')
    plt.savefig('fairness_improved_emergency_category.png', dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

    return final_clusters, df_full, cluster_stats

def summarize_clusters(df_full, cluster_stats):
    """Provide detailed cluster analysis for full dataset."""
    print("\nDETAILED CLUSTER SUMMARY")

    # Overall clustering performance
    total_clustered = len(df_full[df_full['cluster'] != -1])
    clustering_rate = (total_clustered / len(df_full)) * 100

    print(f"\nClustering Performance:")
    print(f"  Total data points: {len(df_full):,}")
    print(f"  Successfully clustered: {total_clustered:,} ({clustering_rate:.1f}%)")
    print(f"  Noise points: {len(df_full) - total_clustered:,} ({100 - clustering_rate:.1f}%)")

    # Cluster quality metrics
    print(f"\nCluster Quality Metrics:")
    valid_clusters = [s for s in cluster_stats if s['cluster'] != -1]

    if valid_clusters:
        avg_size = np.mean([s['size'] for s in valid_clusters])
        size_std = np.std([s['size'] for s in valid_clusters])
        avg_purity = np.mean([s['category_purity'] for s in valid_clusters])

        print(f"  Average cluster size: {avg_size:,.0f} ± {size_std:,.0f}")
        print(f"  Average category purity: {avg_purity:.1f}%")
        print(f"  Size range: {min(s['size'] for s in valid_clusters):,} - {max(s['size'] for s in valid_clusters):,}")

    # Detailed cluster descriptions
    print(f"\nDETAILED CLUSTER DESCRIPTIONS:")

    for stats in cluster_stats:
        cluster_id = stats['cluster']
        cluster_data = df_full[df_full['cluster'] == cluster_id]

        print(f"\nCLUSTER {cluster_id}")

        if cluster_id == -1:
            print(f"NOISE POINTS")
            print(f"   Size: {stats['size']:,} calls ({stats['percentage']:.1f}%)")
            print(f"   Description: Outlier calls that don't fit clear patterns")
        else:
            print(f"STRUCTURED CLUSTER")
            print(f"   Size: {stats['size']:,} calls ({stats['percentage']:.1f}%)")

        print(f"   Time Pattern:")
        print(f"     Peak hour: {stats['avg_hour']:.1f} (spread: ±{stats['std_hour']:.1f}h)")
        print(f"     Peak day: {stats['avg_day']:.1f} (0=Mon, 6=Sun)")
        print(f"   Emergency Type:")
        print(f"     Primary: {stats['top_category']} ({stats['category_purity']:.1f}% purity)")

        if 'avg_lat' in stats:
            print(f"   Geographic Center:")
            print(f"     Location: ({stats['avg_lat']:.4f}, {stats['avg_lng']:.4f})")
            print(f"     Spread: ±{stats['lat_std']:.4f}, ±{stats['lng_std']:.4f}")

        # Additional insights for non-noise clusters
        if cluster_id != -1:
            # Time period classification
            avg_hour = stats['avg_hour']
            if 6 <= avg_hour < 12:
                period = "Morning-focused"
            elif 12 <= avg_hour < 17:
                period = "Afternoon-focused"
            elif 17 <= avg_hour < 21:
                period = "Evening-focused"
            else:
                period = "Night-focused"

            print(f"   Pattern Type: {period}")

            # Category diversity
            cat_counts = cluster_data['category'].value_counts()
            diversity = len(cat_counts)
            print(f"   Category Diversity: {diversity} different types")

            if len(cat_counts) > 1:
                print(f"   Top 3 Categories:")
                for i, (cat, count) in enumerate(cat_counts.head(3).items()):
                    pct = (count / len(cluster_data)) * 100
                    print(f"     {i + 1}. {cat}: {count:,} calls ({pct:.1f}%)")

    # Cross-cluster comparison
    print(f"\nCROSS-CLUSTER COMPARISON:")

    if len(valid_clusters) > 1:
        # Time separation analysis
        print(f"Temporal Separation:")
        for i, cluster1 in enumerate(valid_clusters):
            for cluster2 in valid_clusters[i + 1:]:
                hour_diff = abs(cluster1['avg_hour'] - cluster2['avg_hour'])
                day_diff = abs(cluster1['avg_day'] - cluster2['avg_day'])
                print(f"  Clusters {cluster1['cluster']} vs {cluster2['cluster']}: "
                      f"{hour_diff:.1f}h, {day_diff:.1f} days apart")

        # Category overlap analysis
        print(f"\nCategory Specialization:")
        categories = df_full['category'].unique()
        for category in categories:
            cluster_dist = df_full[df_full['category'] == category]['cluster'].value_counts()
            if len(cluster_dist) > 0:
                primary_cluster = cluster_dist.index[0]
                concentration = (cluster_dist.iloc[0] / cluster_dist.sum()) * 100
                print(f"  {category}: {concentration:.1f}% in Cluster {primary_cluster}")

    return cluster_stats


def perform_time_based_clustering(df, use_full_dataset=True):
    """Perform clustering based on time-based features only - memory optimized for full dataset."""
    print("\nTIME-BASED CLUSTERING")

    print(f"Dataset size: {len(df):,} records")
    df_full = df.copy()
    time_features = ['hour', 'day_of_week_num']

    print(f"Time features: {time_features}")
    print(f"Data shape: {df_full.shape}")

    sample_size = min(50000, len(df_full))  # Use up to 50K for training
    print(f"Training DBSCAN on {sample_size:,} representative sample...")

    # Create stratified sample across time periods
    sample_indices = []
    for hour_group in [(0, 6), (6, 12), (12, 18), (18, 24)]:  # 4 time periods
        hour_mask = (df_full['hour'] >= hour_group[0]) & (df_full['hour'] < hour_group[1])
        hour_data = df_full[hour_mask]
        if len(hour_data) > 0:
            group_sample_size = min(sample_size // 4, len(hour_data))
            if group_sample_size > 0:
                group_indices = np.random.choice(hour_data.index, group_sample_size, replace=False)
                sample_indices.extend(group_indices)

    df_sample = df_full.loc[sample_indices]
    X_time_sample = df_sample[time_features].values.astype(np.float32)

    # Standardize features
    scaler = StandardScaler()
    X_scaled_sample = scaler.fit_transform(X_time_sample)

    # Apply DBSCAN to sample with optimized parameters
    print("Applying DBSCAN to time features sample...")
    dbscan = DBSCAN(
        eps=1.5,  # Increased for fewer clusters
        min_samples=30,  # Increased for fewer, larger clusters
        algorithm='ball_tree', #tree-based data structure that can be efficient for finding nearest neighbors, especially in higher-dimensional spaces
        n_jobs=-1 #number of parallel jobs to run for neighbor search- all available CPU cores will be used
    )

    clusters_sample = dbscan.fit_predict(X_scaled_sample)
    n_sample_clusters = len(set(clusters_sample)) - (1 if -1 in clusters_sample else 0)
    print(f"Found {n_sample_clusters} clusters in sample")

    # Get cluster centroids from sample
    unique_clusters = np.unique(clusters_sample[clusters_sample != -1]) #list of unique cluster labels and filters out the noise points
    cluster_centroids = []

    for cluster_id in unique_clusters:
        cluster_mask = clusters_sample == cluster_id #Boolean mask where data points that belong to the current cluster_id are marked True
        if cluster_mask.sum() > 0:
            centroid = X_scaled_sample[cluster_mask].mean(axis=0)
            cluster_centroids.append(centroid)

    if len(cluster_centroids) > 0:
        cluster_centroids = np.array(cluster_centroids)

        # Apply clustering to full dataset in chunks
        print(f"Applying time clustering to full dataset in chunks...")
        chunk_size = 50000
        clusters_full = []

        for i in range(0, len(df_full), chunk_size):
            chunk_end = min(i + chunk_size, len(df_full))
            print(f"Processing chunk {i // chunk_size + 1}/{(len(df_full) - 1) // chunk_size + 1}...")

            # Process chunk
            chunk_data = df_full.iloc[i:chunk_end][time_features].values.astype(np.float32)
            chunk_scaled = scaler.transform(chunk_data)

            # Find nearest cluster centroid for each point
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=1)
            nbrs.fit(cluster_centroids)
            distances, nearest_indices = nbrs.kneighbors(chunk_scaled)

            # Convert to actual cluster IDs
            chunk_clusters = [unique_clusters[idx[0]] for idx in nearest_indices]
            clusters_full.extend(chunk_clusters)

        clusters = np.array(clusters_full)
    else:
        # Fallback: assign all to noise if no clusters found
        clusters = np.full(len(df_full), -1)

    df_full['time_cluster'] = clusters

    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)

    print(f"Time clustering completed!")
    print(f"Number of time clusters found: {n_clusters}")
    print(f"Number of noise points: {n_noise:,} ({n_noise / len(clusters) * 100:.1f}%)")

    viz_sample_size = min(15000, len(df_full))
    print(f"Creating visualization using {viz_sample_size:,} strategically sampled points...")

    # Stratified sampling by cluster for visualization
    viz_indices = []
    unique_viz_clusters = np.unique(clusters)

    for cluster_id in unique_viz_clusters:
        cluster_mask = clusters == cluster_id
        cluster_size = cluster_mask.sum()

        if cluster_size > 0:
            cluster_sample_size = min(viz_sample_size // len(unique_viz_clusters), cluster_size)
            if cluster_sample_size > 0:
                cluster_indices = np.random.choice(
                    np.where(cluster_mask)[0],
                    cluster_sample_size,
                    replace=False
                )
                viz_indices.extend(cluster_indices)

    df_viz = df_full.iloc[viz_indices]

    plt.figure(figsize=(12, 8))
    scatter = sns.scatterplot(x='hour', y='day_of_week_num', hue='time_cluster',
                              data=df_viz, palette='Set1', alpha=0.6, s=25)
    plt.title(
        f'Time-based DBSCAN Clusters\n(ALL {len(df_full):,} data points, {n_clusters} clusters)\nVisualization: {len(df_viz):,} sample points')
    plt.xlabel('Hour of Day')
    plt.ylabel('Day of Week (0 = Mon, 6 = Sun)')
    plt.grid(True, alpha=0.3)

    # Add legend outside plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig('time_cluster_analysis.png', dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

    # Comprehensive time cluster analysis
    print(f"\nTIME CLUSTER ANALYSIS (Full dataset: {len(df_full):,} records):")
    print("=" * 65)

    time_cluster_counts = df_full['time_cluster'].value_counts().sort_index()
    print("Time cluster sizes:")
    for cluster_id, count in time_cluster_counts.items():
        percentage = (count / len(df_full)) * 100
        print(f"  Time Cluster {cluster_id:2d}: {count:8,d} calls ({percentage:5.1f}%)")

    # Detailed analysis for each time cluster
    time_cluster_stats = []

    for cluster_id in sorted(df_full['time_cluster'].unique()):
        cluster_data = df_full[df_full['time_cluster'] == cluster_id]

        # Time pattern analysis
        hour_mode = cluster_data['hour'].mode().iloc[0] if len(cluster_data) > 0 else 0
        day_mode = cluster_data['day_of_week_num'].mode().iloc[0] if len(cluster_data) > 0 else 0

        # Category analysis
        top_category = cluster_data['category'].mode().iloc[0] if len(cluster_data) > 0 else 'Unknown'
        category_purity = cluster_data['category'].value_counts().iloc[0] / len(cluster_data) * 100 if len(
            cluster_data) > 0 else 0

        # Day name conversion
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_name = day_names[int(day_mode)] if 0 <= day_mode < 7 else 'Unknown'

        stats = {
            'cluster': cluster_id,
            'size': len(cluster_data),
            'percentage': len(cluster_data) / len(df_full) * 100,
            'avg_hour': cluster_data['hour'].mean(),
            'std_hour': cluster_data['hour'].std(),
            'mode_hour': hour_mode,
            'avg_day': cluster_data['day_of_week_num'].mean(),
            'std_day': cluster_data['day_of_week_num'].std(),
            'mode_day': day_mode,
            'day_name': day_name,
            'top_category': top_category,
            'category_purity': category_purity
        }

        time_cluster_stats.append(stats)

        print(f"\n--- Time Cluster {cluster_id} ---")
        print(f"Size: {stats['size']:,} calls ({stats['percentage']:.1f}%)")
        print(f"Peak time: {stats['mode_hour']:02d}:00 on {day_name}")
        print(f"Time spread: Hour {stats['avg_hour']:.1f}±{stats['std_hour']:.1f}")
        print(f"Day spread: {stats['avg_day']:.1f}±{stats['std_day']:.1f}")
        print(f"Dominant category: {top_category} ({category_purity:.1f}% purity)")

        # Show top 3 categories for each cluster
        print("Top categories:")
        top_cats = cluster_data['category'].value_counts(normalize=True).head(3)
        for cat, pct in top_cats.items():
            print(f"  {cat}: {pct * 100:.1f}%")

    chunk_size = 75000
    time_heatmap_chunks = []
    cat_time_chunks = []

    print("Processing data in chunks for heatmap generation...")
    for i in range(0, len(df_full), chunk_size):
        chunk = df_full.iloc[i:i + chunk_size]

        # Time distribution chunk
        time_chunk = pd.crosstab(chunk['day_of_week_num'], chunk['hour'])
        time_heatmap_chunks.append(time_chunk)

        # Category-time cluster chunk
        cat_chunk = pd.crosstab(chunk['category'], chunk['time_cluster'])
        cat_time_chunks.append(cat_chunk)

    # Combine chunks
    time_heatmap = pd.concat(time_heatmap_chunks).groupby(level=0).sum()
    cat_time_combined = pd.concat(cat_time_chunks).groupby(level=0).sum()
    cat_time_cluster = cat_time_combined.div(cat_time_combined.sum(axis=0), axis=1) * 100

    # Create comprehensive time pattern heatmap
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Complete Time Patterns Analysis (ALL {len(df_full):,} calls)', fontsize=16)

    # Overall pattern - all data points
    axes[0, 0].set_title('Complete Time Distribution (All Data)')
    sns.heatmap(time_heatmap, ax=axes[0, 0], cmap='YlOrRd',
                yticklabels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    axes[0, 0].set_xlabel('Hour of Day')
    axes[0, 0].set_ylabel('Day of Week')

    # Time cluster distribution
    axes[0, 1].set_title('Time Cluster Sizes (Complete Dataset)')
    time_cluster_counts.plot(kind='bar', ax=axes[0, 1], color='skyblue')
    axes[0, 1].set_xlabel('Time Cluster ID')
    axes[0, 1].set_ylabel('Number of Calls')
    axes[0, 1].tick_params(axis='x', rotation=0)

    # Category distribution across time clusters
    axes[1, 0].set_title('Categories by Time Cluster (All Data)')
    sns.heatmap(cat_time_cluster, ax=axes[1, 0], annot=True, fmt='.1f', cmap='Blues')
    axes[1, 0].set_xlabel('Time Cluster ID')
    axes[1, 0].set_ylabel('Emergency Category')

    # Hourly distribution by time cluster
    axes[1, 1].set_title('Hourly Distribution by Time Cluster (Complete)')
    for cluster_id in sorted(df_full['time_cluster'].unique())[:8]:  # Show up to 8 clusters
        if cluster_id != -1:  # Skip noise
            cluster_data = df_full[df_full['time_cluster'] == cluster_id]
            hourly_dist = cluster_data['hour'].value_counts().sort_index()
            axes[1, 1].plot(hourly_dist.index, hourly_dist.values,
                            label=f'Cluster {cluster_id}', marker='o', markersize=3)

    axes[1, 1].set_xlabel('Hour of Day')
    axes[1, 1].set_ylabel('Number of Calls')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.savefig('time_cluster.png', dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

    return clusters, df_full, time_cluster_stats

def quick_database_check(db_name='emergency_calls.db'):
    """Quick function to verify database is working and show basic stats."""
    conn = connect_to_database(db_name)
    if conn is None:
        print("Database connection failed")
        return False

    try:
        tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
        tables = pd.read_sql_query(tables_query, conn)
        print(f"Tables found: {', '.join(tables['name'].tolist())}")

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


def run_comprehensive_analysis(db_name="emergency_calls.db", enable_clustering=True,
                               target_pca_clusters=8, target_time_clusters=6, batch_size=50000):
    """Run comprehensive analysis including clustering on full dataset."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE 911 EMERGENCY CALLS ANALYSIS - FULL DATASET CLUSTERING")
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

    # Run analyses that use loaded DataFrame
    daily_results = analyze_daily_patterns(df)
    monthly_results = analyze_monthly_seasonal_patterns(df)
    period_results = analyze_time_periods(df)

    # Create visualizations
    create_visualizations(df)

    # Apply PCA
    pca_df, pca_model, pca_features = apply_pca(df, n_components=0.95)

    clusters = None
    df_clustered = None
    cluster_stats = None
    time_clusters = None
    df_time_clustered = None
    time_cluster_stats = None

    if enable_clustering:
        print(f"Processing all {total_calls:,} records for clustering analysis")
        try:
            # Perform PCA-based clustering on full dataset
            print("PHASE 1: PCA-BASED CLUSTERING")

            # clusters, df_clustered, cluster_stats = perform_dbscan_clustering(df, pca_df, use_full_dataset=True)
            clusters, df_clustered, cluster_stats = perform_dbscan_clustering_fair(df, pca_df, use_full_dataset=True)
            summarize_clusters(df_clustered, cluster_stats)

            # Time-based clustering on full dataset
            print("PHASE 2: TIME-BASED CLUSTERING")

            time_clusters, df_time_clustered, time_cluster_stats = perform_time_based_clustering(df, use_full_dataset=True)

        except MemoryError as e:
            print(f"MEMORY ERROR during clustering: {e}")
            enable_clustering = False

        except Exception as e:
            print(f"CLUSTERING ERROR: {e}")
            enable_clustering = False
    else:
        print("\nCLUSTERING DISABLED")

    # Advanced cross-analysis if clustering succeeded
    if enable_clustering and clusters is not None and time_clusters is not None:
        print("PHASE 3: CROSS-CLUSTERING ANALYSIS")

        # Compare PCA clusters vs Time clusters
        print("Comparing PCA-based vs Time-based clustering results...")

        # Create cross-tabulation
        cross_cluster = pd.crosstab(df_clustered['cluster'], df_time_clustered['time_cluster'],
                                    margins=True, margins_name="Total")

        print(f"\nCross-tabulation of PCA Clusters vs Time Clusters:")
        print(cross_cluster)

        # Calculate agreement metrics
        # Remove noise points for agreement calculation
        mask = (df_clustered['cluster'] != -1) & (df_time_clustered['time_cluster'] != -1)
        if mask.sum() > 0:
            # Simple agreement measure
            agreement_data = df_clustered[mask].copy()
            agreement_data['time_cluster'] = df_time_clustered[mask]['time_cluster'].values

            # Find dominant time cluster for each PCA cluster
            for pca_cluster in sorted(agreement_data['cluster'].unique()):
                pca_data = agreement_data[agreement_data['cluster'] == pca_cluster]
                time_dist = pca_data['time_cluster'].value_counts()
                if len(time_dist) > 0:
                    dominant_time = time_dist.index[0]
                    dominance_pct = (time_dist.iloc[0] / len(pca_data)) * 100
                    print(f"  PCA Cluster {pca_cluster} → Time Cluster {dominant_time} ({dominance_pct:.1f}%)")


    if hourly_results:
        peak_hour = hourly_results['hourly_counts'].idxmax()
        peak_count = hourly_results['hourly_counts'].max()
        evening_pct = hourly_results['evening_percentage']

        print(f"TEMPORAL PATTERNS:")
        print(f"   Peak hour: {peak_hour}:00 with {int(peak_count):,d} calls")
        print(f"   Evening concentration: {evening_pct:.1f}% of all calls")

    if daily_results:
        busiest_day = daily_results['daily_counts'].idxmax()
        weekday_avg = daily_results['weekday_count'] / 5
        weekend_avg = daily_results['weekend_count'] / 2

        print(f"   Busiest day: {busiest_day}")
        print(f"   Weekday average: {weekday_avg:,.0f} calls/day")
        print(f"   Weekend average: {weekend_avg:,.0f} calls/day")

    if enable_clustering and clusters is not None:
        unique_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        noise_points = list(clusters).count(-1)
        clustering_rate = ((len(clusters) - noise_points) / len(clusters)) * 100

        print(f"PCA CLUSTERING RESULTS:")
        print(f"   Clusters identified: {unique_clusters}")
        print(f"   Successfully clustered: {clustering_rate:.1f}% of data")
        print(f"   Noise points: {noise_points:,}")

        if time_clusters is not None:
            unique_time_clusters = len(set(time_clusters)) - (1 if -1 in time_clusters else 0)
            time_noise_points = list(time_clusters).count(-1)
            time_clustering_rate = ((len(time_clusters) - time_noise_points) / len(time_clusters)) * 100

            print(f" TIME CLUSTERING RESULTS:")
            print(f"   Time-based clusters: {unique_time_clusters}")
            print(f"   Time clustering rate: {time_clustering_rate:.1f}%")
            print(f"   Time noise points: {time_noise_points:,}")

    print(f"\nANALYSIS COMPLETED SUCCESSFULLY!")
    print(f"   Dataset size: {total_calls:,} emergency calls")
    print(f"   Clustering: {'Full dataset analysis' if enable_clustering else 'Disabled for memory optimization'}")

    return {
        'total_calls': total_calls,
        'hourly': hourly_results,
        'daily': daily_results,
        'monthly': monthly_results,
        'periods': period_results,
        'database_stats': summary_stats,
        'clusters': clusters,
        'cluster_stats': cluster_stats,
        'time_clusters': time_clusters,
        'time_cluster_stats': time_cluster_stats,
        'pca_components': pca_df,
        'clustered_data': df_clustered,
        'time_clustered_data': df_time_clustered,
        'clustering_enabled': enable_clustering,
        'target_pca_clusters': target_pca_clusters,
        'target_time_clusters': target_time_clusters
    }


if __name__ == "__main__":
    db_name = 'emergency_calls.db'
    csv_file = '911.csv'

    # Configuration options
    ENABLE_CLUSTERING = True  # Set to False to skip clustering for very large datasets
    CLUSTERING_SAMPLE_SIZE = 50000  # Adjust based on available memory

    # Check if database exists, if not create it from CSV
    if not os.path.exists(db_name):
        if os.path.exists(csv_file):
            print(f"Database not found. Creating from {csv_file}...")
            conn, df_clean = setup_database_from_csv(csv_file, db_name)
            conn.close()
        else:
            print(f"Error: Neither database ({db_name}) nor CSV file ({csv_file}) found")
            print("Available files:", [f for f in os.listdir('.') if f.endswith(('.csv', '.db'))])
            exit(1)

    # Quick database check
    if not quick_database_check(db_name):
        exit(1)

    print(f"Starting comprehensive analysis using database...")

    if ENABLE_CLUSTERING:
        print(f"Clustering enabled with sample size: {CLUSTERING_SAMPLE_SIZE:,}")
    else:
        print("Clustering disabled for memory optimization")

    try:
        # Run comprehensive analysis with clustering
        results = run_comprehensive_analysis(db_name,enable_clustering=ENABLE_CLUSTERING)

        print(f"\nANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"All data sourced from database: {db_name}")

        if results:
            print(f"\nQUICK SUMMARY:")

            if 'hourly' in results and results['hourly']:
                hourly = results['hourly']
                print(f"Peak activity hour: {hourly['hourly_counts'].idxmax()}:00")
                print(f"Evening concentration: {hourly['evening_percentage']:.1f}%")

            if 'daily' in results and results['daily']:
                daily = results['daily']
                print(f"Busiest day of week: {daily['daily_counts'].idxmax()}")

            if ENABLE_CLUSTERING and 'clusters' in results and results['clusters'] is not None:
                clusters = results['clusters']
                unique_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
                noise_points = list(clusters).count(-1)
                print(f"Clustering: {unique_clusters} clusters, {noise_points} noise points")

                if 'time_clusters' in results and results['time_clusters'] is not None:
                    time_clusters = results['time_clusters']
                    unique_time_clusters = len(set(time_clusters)) - (1 if -1 in time_clusters else 0)
                    time_noise_points = list(time_clusters).count(-1)
                    print(f"Time clustering: {unique_time_clusters} clusters, {time_noise_points} noise points")

    except MemoryError:
        print("\nMEMORY ERROR: Dataset too large for available memory")
        print("Try reducing CLUSTERING_SAMPLE_SIZE or setting ENABLE_CLUSTERING = False")
    except FileNotFoundError:
        print("Error: Required files not found")
        print("Make sure both the database and any required Python packages are available")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback

        traceback.print_exc()