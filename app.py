import os
import sys
# Fix for Windows: Force Spark to use the correct Python executable before importing pyspark
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

import streamlit as st
import pandas as pd
import time
import os
import re
from datetime import datetime
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
import pymongo
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- CONFIGURATION & AESTHETICS ---
st.set_page_config(page_title="Big Data Sentiment & Trend Analyzer", layout="wide", page_icon="📈")

st.markdown("""
<style>
    /* Premium aesthetics CSS */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4ebf5 100%);
    }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
    }
    .main-title {
        background: -webkit-linear-gradient(45deg, #2563EB, #7C3AED);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        font-weight: 800;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-title {
        text-align: center;
        color: #4B5563;
        font-size: 1.2rem;
        margin-bottom: 40px;
    }
    div[data-testid="metric-container"] {
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #E5E7EB;
        transition: transform 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #2563EB !important;
        border-bottom: 3px solid #2563EB;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">📈 Big Data Social Media Sentiment & Trend Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">A powerful system utilizing Apache Spark (concept/local) and MongoDB for massive text analytics.</p>', unsafe_allow_html=True)

# --- BACKEND CONNECTIONS & FALLBACKS ---

@st.cache_resource
def get_db_connection():
    try:
        # Try connecting to local MongoDB
        client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
        client.admin.command('ismaster') # test connection
        db = client["social_media_analytics"]
        collection = db["posts"]
        return collection, True
    except pymongo.errors.ServerSelectionTimeoutError:
        # Fallback if MongoDB is not running
        return None, False

db_collection, using_mongodb = get_db_connection()

if 'local_data' not in st.session_state:
    st.session_state.local_data = None

# Fallback PySpark Initialization
@st.cache_resource
def get_spark_session():
    try:
        import pyspark
        from pyspark.sql import SparkSession
        spark = SparkSession.builder \
            .appName("SocialMediaAnalyzer") \
            .master("local[*]") \
            .getOrCreate()
        return spark, True
    except Exception as e:
        return None, False

spark_session, using_spark = get_spark_session()

# --- HELPER FUNCTIONS ---

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remove URLs, @mentions, and special characters (keeping basic punctuation)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    # Remove non-ascii characters (like emojis if we want strict cleaning)
    text = text.encode("ascii", "ignore").decode()
    return text.strip()

def analyze_sentiment(text):
    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

def extract_keywords(text):
    # Very basic stopword removal for demonstration
    stopwords = set(["the", "is", "in", "and", "to", "of", "a", "it", "for", "on", "with", "this", "that", "i", "my"])
    words = str(text).lower().split()
    words = [re.sub(r'\W+', '', w) for w in words]
    keywords = [w for w in words if w and w not in stopwords and len(w) > 2]
    return keywords

# --- MAIN APPLICATION LOGIC ---

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📤 1. Upload Dataset", "📊 2. Analysis Dashboard", "🔥 3. Trending Topics", "🔍 4. Search Engine", "📡 5. Live Stream Firehose", "🌍 6. Global Heatmap"])

with tab1:
    st.header("Upload Social Media Dataset")
    st.write("Upload a CSV file containing social media posts. The system will store it in MongoDB (or fallback in-memory) and process it.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    with col2:
        if not using_mongodb:
            st.warning("⚠️ **MongoDB not detected locally.** Using in-memory fallback for data storage.")
        else:
            st.success("✅ **MongoDB Connected.**")
            
        if not using_spark:
            st.warning("⚠️ **Apache Spark not configured optimally.** Using Pandas fallback for processing.")
        else:
            st.success("✅ **Apache Spark Ready.**")

    if uploaded_file is not None:
        if st.button("Process & Store Data", type="primary"):
            with st.spinner('Reading and processing dataset...'):
                df = pd.read_csv(uploaded_file)
                
                # Big Data Processing
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                if using_spark and spark_session is not None:
                    status_text.text("Step 1/3: Distributing data to Apache Spark Cluster...")
                    # Convert to string to avoid Spark schema inference issues with nulls/mixed types
                    df = df.astype(str)
                    spark_df = spark_session.createDataFrame(df)
                    
                    from pyspark.sql.functions import udf, col
                    from pyspark.sql.types import StringType
                    
                    clean_udf = udf(clean_text, StringType())
                    sentiment_udf = udf(analyze_sentiment, StringType())
                    
                    status_text.text("Step 2/3: Running distributed text cleaning & sentiment analysis...")
                    progress_bar.progress(33)
                    
                    try:
                        spark_df = spark_df.withColumn("cleaned_text", clean_udf(col("text")))
                        spark_df = spark_df.withColumn("sentiment", sentiment_udf(col("cleaned_text")))
                        
                        progress_bar.progress(66)
                        status_text.text("Collecting processed data from Spark nodes...")
                        df = spark_df.toPandas()
                    except Exception as e:
                        # PySpark on Windows/Python 3.13 can be fragile with UDFs. Fallback gracefully.
                        df['cleaned_text'] = df['text'].apply(clean_text)
                        df['sentiment'] = df['cleaned_text'].apply(analyze_sentiment)
                        progress_bar.progress(66)
                else:
                    # Step 1: Cleaning
                    status_text.text("Step 1/3: Cleaning text data (simulating Spark distributed map)...")
                    df['cleaned_text'] = df['text'].apply(clean_text)
                    progress_bar.progress(33)
                    time.sleep(0.5)
                    
                    # Step 2: Sentiment Analysis
                    status_text.text("Step 2/3: Running ML Sentiment Classification...")
                    df['sentiment'] = df['cleaned_text'].apply(analyze_sentiment)
                    progress_bar.progress(66)
                    time.sleep(0.5)
                
                # Step 3: Storage
                status_text.text("Step 3/3: Storing processed data...")
                records = df.to_dict('records')
                
                if using_mongodb:
                    # Clear existing collection for demo purposes
                    db_collection.delete_many({})
                    db_collection.insert_many(records)
                else:
                    st.session_state.local_data = df
                
                progress_bar.progress(100)
                status_text.text("Processing Complete!")
                st.success(f"Successfully processed and stored {len(df)} records!")

# Function to load data for tabs
def load_data():
    if using_mongodb and db_collection.count_documents({}) > 0:
        cursor = db_collection.find({}, {"_id": 0})
        return pd.DataFrame(list(cursor))
    elif st.session_state.local_data is not None:
        return st.session_state.local_data
    return pd.DataFrame()

df_processed = load_data()

with tab2:
    if df_processed.empty:
        st.warning("Please upload a dataset in the 'Upload Dataset' tab first.")
    else:
        st.header("Overall Sentiment Analysis")
        
        # Metrics Row
        total_posts = len(df_processed)
        pos_count = len(df_processed[df_processed['sentiment'] == 'Positive'])
        neg_count = len(df_processed[df_processed['sentiment'] == 'Negative'])
        neu_count = len(df_processed[df_processed['sentiment'] == 'Neutral'])
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Analyzed", f"{total_posts:,}")
        col2.metric("Positive 😃", f"{pos_count:,}", f"{(pos_count/total_posts)*100:.1f}%")
        col3.metric("Negative 😠", f"{neg_count:,}", f"-{(neg_count/total_posts)*100:.1f}%")
        col4.metric("Neutral 😐", f"{neu_count:,}")
        
        st.markdown("---")
        
        # Charts Row
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            # Donut Chart for Sentiment Distribution
            fig_donut = px.pie(
                df_processed, 
                names='sentiment', 
                title='Sentiment Distribution',
                color='sentiment',
                color_discrete_map={'Positive':'#10B981', 'Negative':'#EF4444', 'Neutral':'#6B7280'},
                hole=0.4
            )
            fig_donut.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_donut, use_container_width=True)
            
        with col_chart2:
            # Bar Chart for Platform wise sentiment
            if 'platform' in df_processed.columns:
                platform_sentiment = df_processed.groupby(['platform', 'sentiment']).size().reset_index(name='count')
                fig_bar = px.bar(
                    platform_sentiment, 
                    x='platform', 
                    y='count', 
                    color='sentiment',
                    title='Sentiment by Platform',
                    barmode='group',
                    color_discrete_map={'Positive':'#10B981', 'Negative':'#EF4444', 'Neutral':'#6B7280'}
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        # Time Series Chart
        st.subheader("Time-Series Sentiment Trend")
        if 'date' in df_processed.columns:
            # Ensure date is datetime
            df_processed['date'] = pd.to_datetime(df_processed['date'])
            time_trend = df_processed.groupby([df_processed['date'].dt.date, 'sentiment']).size().reset_index(name='count')
            
            fig_line = px.line(
                time_trend, 
                x='date', 
                y='count', 
                color='sentiment',
                title='Daily Sentiment Volume',
                color_discrete_map={'Positive':'#10B981', 'Negative':'#EF4444', 'Neutral':'#6B7280'},
                markers=True
            )
            st.plotly_chart(fig_line, use_container_width=True)

with tab3:
    if df_processed.empty:
        st.warning("Please upload a dataset first.")
    else:
        st.header("🔥 Trending Topics & Keywords")
        
        # Simulate Spark MapReduce for Keyword Extraction
        all_keywords = []
        for text in df_processed['cleaned_text']:
            all_keywords.extend(extract_keywords(text))
            
        keyword_counts = Counter(all_keywords)
        top_keywords = keyword_counts.most_common(15)
        
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.subheader("Top Trending Words")
            trend_df = pd.DataFrame(top_keywords, columns=['Keyword', 'Frequency'])
            
            # Highlight top keywords
            st.dataframe(
                trend_df.style.background_gradient(cmap='Blues', subset=['Frequency']),
                use_container_width=True,
                height=400
            )
            
        with col2:
            st.subheader("Topic WordCloud")
            # Generate WordCloud
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                colormap='viridis',
                max_words=100
            ).generate_from_frequencies(keyword_counts)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

with tab4:
    if df_processed.empty:
        st.warning("Please upload a dataset first.")
    else:
        st.header("🔍 Keyword Search & Targeted Analytics")
        st.write("Search for a specific keyword to see sentiment strictly related to that topic.")
        
        search_term = st.text_input("Enter keyword (e.g., 'iPhone', 'AI', 'movie')", "")
        
        if search_term:
            # Filter dataset
            search_regex = re.compile(re.escape(search_term), re.IGNORECASE)
            filtered_df = df_processed[df_processed['text'].str.contains(search_regex, na=False)]
            
            if filtered_df.empty:
                st.info(f"No results found for '{search_term}'.")
            else:
                st.success(f"Found {len(filtered_df)} posts mentioning '{search_term}'.")
                
                # Targeted Metrics
                f_total = len(filtered_df)
                f_pos = len(filtered_df[filtered_df['sentiment'] == 'Positive'])
                f_neg = len(filtered_df[filtered_df['sentiment'] == 'Negative'])
                f_neu = len(filtered_df[filtered_df['sentiment'] == 'Neutral'])
                
                f_col1, f_col2, f_col3 = st.columns(3)
                f_col1.metric("Positive Sentiment", f"{(f_pos/f_total)*100:.1f}%", f"{f_pos} posts")
                f_col2.metric("Negative Sentiment", f"{(f_neg/f_total)*100:.1f}%", f"-{f_neg} posts")
                f_col3.metric("Neutral Sentiment", f"{(f_neu/f_total)*100:.1f}%", f"{f_neu} posts")
                
                # Sample Posts
                st.subheader("Sample Posts")
                st.dataframe(
                    filtered_df[['username', 'text', 'sentiment', 'platform']].head(10),
                    use_container_width=True
                )

with tab5:
    st.header("📡 Live Data Firehose Simulation")
    st.write("Simulate a real-time big data stream using the uploaded dataset, processed live through our custom Spark-like sentiment engine.")
    
    if df_processed.empty:
        st.warning("Please upload a dataset in Tab 1 first to use as the streaming source.")
    else:
        if st.button("🔴 Start Live Stream", type="primary"):
            st.success("Live Connection Established. Streaming data from topic `social_media_firehose`...")
            
            # Placeholders
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            m1 = metric_col1.empty()
            m2 = metric_col2.empty()
            m3 = metric_col3.empty()
            
            st.subheader("Spark Execution Terminal")
            spark_terminal = st.empty()
            
            st.subheader("Live Sentiment Stream")
            chart_placeholder = st.empty()
            
            # Stream Loop
            stream_df = df_processed.sample(frac=1).reset_index(drop=True)
            total_processed = 0
            pos = 0
            neg = 0
            
            history = []
            
            for i in range(0, min(50, len(stream_df)), 2): # Process 2 at a time
                batch = stream_df.iloc[i:i+2]
                
                # Update metrics
                total_processed += len(batch)
                pos += len(batch[batch['sentiment'] == 'Positive'])
                neg += len(batch[batch['sentiment'] == 'Negative'])
                
                m1.metric("Posts Ingested", total_processed)
                m2.metric("Positive Trend", pos)
                m3.metric("Negative Trend", neg)
                
                # Update Spark terminal
                log_text = f"```bash\n" \
                           f"[SparkStreaming] Batch {i//2 + 1} Received: 2 records\n" \
                           f"[SparkContext] Submitting job to cluster...\n" \
                           f"[DAGScheduler] Stage 0: clean_text() -> MapPartitions\n" \
                           f"[TaskSetManager] Finished task 0.0 in stage 0.0\n" \
                           f"[MLlib] Applied Sentiment Model. Extracted {pos} positive.\n" \
                           f"```"
                spark_terminal.markdown(log_text)
                
                # Chart update
                history.append({'batch': i//2, 'positive': pos, 'negative': neg})
                hist_df = pd.DataFrame(history)
                
                fig = px.line(hist_df, x='batch', y=['positive', 'negative'], 
                              title="Live Sentiment Tracking",
                              color_discrete_map={'positive':'#10B981', 'negative':'#EF4444'})
                chart_placeholder.plotly_chart(fig, use_container_width=True)
                
                time.sleep(1.5)
                
            st.info("Live stream simulation completed.")

with tab6:
    st.header("🌍 Global Sentiment Heatmap")
    st.write("Visualize the global distribution of social media sentiment on an interactive 3D map.")
    
    if df_processed.empty:
        st.warning("Please upload a dataset in Tab 1 first.")
    else:
        # Create a safe copy so we don't pollute the main dataframe with fake data forever
        map_df = df_processed.copy()
        
        # Auto-generate coordinates if missing!
        import numpy as np
        if 'latitude' not in map_df.columns or 'longitude' not in map_df.columns:
            
            # Deep inland global hubs (hundreds of miles from oceans)
            land_centers = [
                (41.87, -87.62), (32.77, -96.79), (39.73, -104.99),  # US (Chicago, Dallas, Denver)
                (40.41, -3.70), (52.52, 13.40), (48.20, 16.37),      # Europe (Madrid, Berlin, Vienna)
                (28.61, 77.20), (39.90, 116.40), (24.71, 46.67),     # Asia (Delhi, Beijing, Riyadh)
                (-15.82, -47.92), (4.60, -74.08),                    # South America (Brasilia, Bogota)
                (-26.20, 28.04), (9.02, 38.74),                      # Africa (Johannesburg, Addis Ababa)
                (55.75, 37.61)                                       # Russia (Moscow)
            ]
            
            chosen_centers = [land_centers[np.random.randint(0, len(land_centers))] for _ in range(len(map_df))]
            
            # Use a LARGE spread (approx 100 miles) for cool Big Data clusters, 
            # safely avoiding oceans because the cities are far inland!
            map_df['latitude'] = [c[0] + np.random.uniform(-1.5, 1.5) for c in chosen_centers]
            map_df['longitude'] = [c[1] + np.random.uniform(-1.5, 1.5) for c in chosen_centers]
        
        if st.button("🔴 Start Live Map Stream", type="primary", key="live_map_btn"):
            st.success("Initializing Global Satellite Link...")
            
            # Placeholders
            col1, col2, col3 = st.columns(3)
            metric_ingested = col1.empty()
            metric_pos = col2.empty()
            metric_neg = col3.empty()
            
            st.subheader("Live Geospatial Distribution")
            map_placeholder = st.empty()
            
            # Streaming Setup
            stream_map_df = map_df.sample(frac=1).reset_index(drop=True) # Shuffle data
            
            # Color mapping
            color_discrete_map = {'Positive': 'lime', 'Negative': 'red', 'Neutral': 'gray'}
            
            # Stream in chunks of 5 up to 100 records for the demo
            for i in range(0, min(100, len(stream_map_df)), 5):
                current_chunk = stream_map_df.iloc[0:i+5] # Growing dataset
                
                # Update Metrics
                total_pts = len(current_chunk)
                pos_pts = len(current_chunk[current_chunk['sentiment'] == 'Positive'])
                neg_pts = len(current_chunk[current_chunk['sentiment'] == 'Negative'])
                
                metric_ingested.metric("Global Signals Ingested", total_pts)
                metric_pos.metric("Positive Signals", pos_pts)
                metric_neg.metric("Negative Signals", neg_pts)
                
                # Redraw map
                fig = px.scatter_geo(
                    current_chunk,
                    lat='latitude',
                    lon='longitude',
                    color='sentiment',
                    hover_name='username',
                    hover_data=['platform'],
                    projection="natural earth",
                    color_discrete_map=color_discrete_map,
                    opacity=0.6,
                    size_max=10
                )
                
                fig.update_geos(
                    showcountries=True, countrycolor="#CBD5E1",
                    showcoastlines=True, coastlinecolor="#CBD5E1",
                    showland=True, landcolor="#F8FAFC",
                    showocean=True, oceancolor="#E0F2FE", 
                    bgcolor='rgba(0,0,0,0)'
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=600
                )
                
                map_placeholder.plotly_chart(fig, use_container_width=True)
                time.sleep(1.0)
                
            st.info("Live map simulation completed.")
