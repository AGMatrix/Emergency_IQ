README.TXT
===========

PROJECT TITLE:
---------------
EmergencyIQ: A Predictive Analytics Platform for Optimizing Emergency Response Resource Allocation Using 911 Call Data

DESCRIPTION:
-------------
This project performs a comprehensive temporal, spatial, and fairness-focused analysis of a 911 emergency call dataset with full database integration. It includes preprocessing, missing value handling, exploratory data analysis (EDA), visualization, dimensionality reduction using PCA, fairness testing, database creation, entity-relationship (ER) diagram generation, and advanced SQL-based analytics.

WHAT HAS BEEN DONE:
---------------------

 1. Data Loading & Preprocessing:
   - Loaded 911 call data (first 10000 records) into structured SQLite database
   - Parsed and extracted temporal features: hour, day of week, month, season, and time period
   - Categorized emergency types into EMS, Fire, and Traffic
   - Implemented robust data cleaning and standardization pipelines

 2. Database Architecture & Management:
   - Database Schema Design:Created normalized relational schema with proper indexing
   - Automated Database Creation: Complete pipeline from CSV to structured SQLite database
   - Data Structure Inspection: Comprehensive analysis of raw CSV structure
   - Summary Table Generation:** Automated creation of analytical tables:
     - `emergency_calls`: Main table with all call records and derived features
     - `zip_code_stats`: Aggregated statistics by ZIP code
     - `temporal_patterns`: Time-based pattern analysis (hourly, daily, monthly, seasonal)
     - `emergency_types`: Category-based statistics and distributions
   - Database Testing Suite:** Comprehensive query testing and validation

 3. Enhanced Analysis Functions:
   - Database-Integrated Temporal Analysis:
     - `analyze_hourly_patterns_db()`: Peak hours analysis with evening concentration metrics
     - `analyze_daily_patterns()`: Weekday vs weekend pattern analysis
     - `analyze_monthly_seasonal_patterns()`: Monthly and seasonal trend analysis
     - `analyze_time_periods()`: Time period distribution analysis
   - Geospatial Analysis:
     - `analyze_geospatial_db()`: ZIP code analysis, township analysis, coordinate availability
     - Geographic hotspot identification with coordinate statistics
   - Missing Value & Outlier Analysis:
     - `analyze_missing_values_db()`: Database-based missing value analysis
     - `detect_outliers_db()`: Statistical outlier detection with IQR methods
     - `analyze_missing_zip_codes_db()`: ZIP code recovery analysis

 4. Advanced Visualization System:
   - Temporal Visualizations:
     - `create_visualizations()`: Comprehensive temporal analysis plots
     - Hourly distribution with peak hour highlighting
     - Daily, monthly, and seasonal distribution analysis
     - Time period distribution with color-coded periods
     - Category distribution by hour (heatmap style)
   - Bivariate Analysis:
     - `bivariate_analysis()`: Category vs day of week analysis
     - Cross-tabulation with percentage distributions
   - Correlation Analysis:
     - `correlation_heatmap()`: Numeric feature correlation matrix
     - Advanced heatmap visualization with color coding

 5. Time Series & Advanced Analytics:
   - Time Series Analysis:
     - `plot_time_series()`: Daily call volume trends
     - December-focused analysis with trend visualization
   - Principal Component Analysis (PCA):
     - `apply_pca()`: Dimensionality reduction with 95% variance preservation
     - `explain_pca_components()`: Component interpretation and loading analysis
     - Feature standardization and component visualization
   - Fairness & Bias Testing:
     - `test_pca_correlation_with_sensitive_attrs()`: Correlation analysis with sensitive attributes
     - `run_fairness_tests()`: Kruskal-Wallis fairness testing
     - `visualize_pca_bias()`: PCA component distribution visualization by groups

 6. Advanced Clustering Analysis:
   - DBSCAN Clustering on PCA Components:
     - K-distance graph analysis for optimal epsilon parameter selection
     - DBSCAN clustering using first 4 principal components (PC1-PC4)
     - Cluster visualization on PC3 vs PC4 scatter plots
     - Standardized feature scaling for clustering optimization
   - Cluster Characterization:
     - `summarize_clusters()`: Comprehensive cluster analysis function
     - Cluster size distribution and statistical summaries
     - Geographic and temporal pattern analysis per cluster
     - Emergency category distribution within each cluster
     - Average location coordinates and time patterns per cluster
   - Temporal Clustering:
     - Time-based DBSCAN clustering using hour and day-of-week features
     - Temporal pattern visualization with hour vs day-of-week scatter plots
     - Time cluster characterization with average temporal features
     - Category distribution analysis within temporal clusters
   - Cluster Visualization:
     - Multi-dimensional cluster scatter plots with color coding
     - Hourly activity patterns by cluster using count plots
     - Time-based cluster visualization with grid overlays

 7. SQL Query Demonstration System:
   - Advanced SQL Analytics:
     - `demonstrate_sql_queries()`: Comprehensive SQL query demonstrations
     - Peak hours analysis by emergency type
     - Monthly trends analysis
     - Geographic hotspot identification
     - Weekend vs weekday pattern analysis
     - Seasonal emergency pattern analysis
   - Database Verification:
     - `quick_database_check()`: Database integrity verification
     - Table existence checking and basic statistics

 8. Comprehensive Reporting System:
   - Integrated Report Generation:
     - `generate_comprehensive_report_db()`: Complete analysis pipeline
     - Database-sourced analytics with summary statistics
     - Key findings extraction and interpretation
     - Multi-dimensional analysis integration
   - Main Execution Pipeline:
     - Database connectivity verification
     - Sequential analysis execution
     - Error handling and progress tracking
     - Summary statistics generation

 9. Database ER Diagram & Documentation:
   - Created programmatic ER diagram of database schema
   - Shows key entity tables, attributes, and relationships (1:M)
   - Saved diagram as both PNG and PDF formats

WHAT CAN BE DONE (Optional Extensions):
----------------------------------------
- Apply reverse geocoding to recover missing ZIP codes using lat/lng coordinates
- Develop interactive dashboard using Plotly or Dash for real-time analysis
- Implement machine learning models for call volume prediction and emergency type classification
- Add spatial clustering analysis with geographic heat maps and density analysis

HOW FUTURE EXTENSIONS WILL BE IMPLEMENTED:
--------------------------------------------
- Continue building on the current modular Python codebase
- Use scikit-learn pipelines for predictive modeling
- Use K-Means Clustering to group latitude/longitude points into fixed-size clusters
- Use Rule-based Feature Engineering  or Natural Language processing to extract context from messy location descriptions
- Use NumPy/Pandas to compute call frequency by hour/day for each cluster

TECHNICAL COVERAGE:
--------------------
- Data Engineering: Database design, ETL pipelines, data quality management
- Data Analysis: Statistical analysis, temporal patterns, geospatial analysis
- Visualization: Multi-dimensional plotting, heatmaps, time series visualization
- Machine Learning: PCA, dimensionality reduction, fairness testing
- Clustering Analysis: K-distance analysis, temporal clustering, cluster characterization
- Database Management: SQL querying, schema design, performance optimization
- Bias Detection: Statistical fairness testing, correlation analysis
- Reporting: Automated report generation, summary statistics

DELIVERABLES:
--------------

 1. Database Components:
   - SQLite database file: `emergency_calls.db`
   - Normalized schema with 4 tables and proper indexing
   - Automated data preparation and cleaning pipeline
   - Database testing and validation suite

 2. Analysis Scripts:
   - Database-integrated temporal analysis functions
   - Geospatial analysis with coordinate statistics
   - Missing value and outlier detection algorithms
   - PCA implementation with fairness testing
   - SQL query demonstration system

 3. Visualization Outputs:
   - `temporal_analysis_911_calls_from_db.png`: Comprehensive temporal analysis
   - `bivariate_category_day_from_db.png`: Category vs day analysis
   - `correlation_heatmap_from_db.png`: Numeric feature correlations
   - `pca_component_loadings_from_db.png`: PCA component analysis
   - Individual PCA component boxplots by category groups
   - K-distance graphs for DBSCAN parameter optimization
   - DBSCAN cluster scatter plots (PC3 vs PC4)
   - Temporal cluster visualizations (hour vs day-of-week)
   - Hourly activity patterns by cluster

 4. Documentation:
   - ER Diagram files: `er_diagram.png` and `er_diagram.pdf`
   - Database schema documentation
   - Comprehensive analysis reports
   - SQL query examples and demonstrations

 5. Advanced Analytics:
   - PCA-transformed datasets with component interpretation
   - Fairness testing results and statistical analysis
   - Time series analysis with trend identification
   - Geographic hotspot analysis with coordinate data
   - DBSCAN clustering results with cluster characterization
   - Temporal clustering analysis with time-based patterns
   - K-distance analysis for optimal clustering parameters

INSTALLATION & EXECUTION INSTRUCTIONS:
---------------------------------------

Requirements:
- Python 3.7+
- Required packages:
  ```
  pandas>=1.3.0
  numpy>=1.21.0
  matplotlib>=3.5.0
  seaborn>=0.11.0
  scikit-learn>=1.0.0
  scipy>=1.7.0
  openpyxl>=3.0.0
  ```
- SQLite3 (included with Python standard library)

Installation:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy openpyxl
```

Execution Steps:

1. Database Setup (First Time):
```bash
python phase3.py
```
This will automatically:
- Inspect the raw CSV data structure
- Create the normalized database schema
- Clean and prepare the data
- Load data into the database
- Generate summary tables
- Run validation tests

- Run Complete Analysis:

  This executes:
  - Database connectivity verification
  - Comprehensive temporal analysis
  - Geospatial analysis
  - Missing value and outlier detection
  - PCA and fairness testing
  - Visualization generation
  - SQL query demonstrations
  - Report generation
- Advanced Clustering Analysis:**
```python
# PCA-based clustering
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import numpy as np

# Extract PCA components for clustering
X = pca_df[['PC1', 'PC2', 'PC3', 'PC4']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-distance analysis for parameter selection
min_samples = 5
neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)

# DBSCAN clustering
dbscan = DBSCAN(eps=1.2, min_samples=min_samples)
clusters = dbscan.fit_predict(X_scaled)

# Temporal clustering
time_features = ['hour', 'day_of_week_num']
X_time = df[time_features]
X_time_scaled = scaler.fit_transform(X_time)
dbscan_time = DBSCAN(eps=0.8, min_samples=10)
time_clusters = dbscan_time.fit_predict(X_time_scaled)
```
2. Generate Visualizations:
```bash
python analysis_database_integrated.py
```


Database Structure:
- emergency_calls: Main table with all call records, temporal features, and categories
- zip_code_stats: Aggregated statistics by ZIP code including peak hours and dominant categories
- temporal_patterns: Time-based pattern analysis with hourly, daily, monthly, and seasonal distributions
- emergency_types: Category-based statistics with call volumes and percentages

Key Features:
- Database Integration: All analysis functions work directly with SQLite database
- Automated Data Quality: Comprehensive validation and cleaning pipelines
- Performance Optimization: Indexed database schema for fast queries
- Modular Design: Easy to extend and maintain codebase
- Advanced Analytics: PCA, fairness testing, and statistical analysis
- Visualizations: Multi-dimensional plotting with publication-quality outputs
- SQL Demonstrations: Complex analytical queries with practical examples

Output Files Generated:
- Database: `emergency_calls.db`
- Visualizations: `*_from_db.png` files
- ER Diagrams: `er_diagram.png`, `er_diagram.pdf`
- Clustering visualizations: K-distance plots, cluster scatter plots
- Analysis reports: Console output with detailed statistics

Error Handling:
- Database connection validation
- Missing file detection
- Data quality checks
- Graceful error reporting with detailed messages
- Automatic fallback mechanisms

Performance Notes:
- Database queries are optimized with proper indexing
- Memory-efficient data loading for large datasets
- Cached summary statistics for improved performance
- Modular design allows selective analysis execution
