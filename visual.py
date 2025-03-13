# Voice to Visualization System
# This implementation includes:
# 1. Voice Recognition
# 2. Natural Language to SQL Query Conversion
# 3. Database Connection and Query Execution
# 4. Data Visualization

import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import speech_recognition as sr
from sqlalchemy import create_engine
import numpy as np
import json
from transformers import pipeline
import sqlite3
import plotly.express as px
import plotly.graph_objects as go

class VoiceToVisualization:
    def __init__(self, db_connection_string=None):
        """Initialize the Voice to Visualization system"""
        # If no connection string is provided, use a sample SQLite database
        if db_connection_string is None:
            self.create_sample_database()
            self.db_connection_string = "sqlite:///sample_business_data.db"
        else:
            self.db_connection_string = db_connection_string
        
        # Initialize the NLP pipeline for query generation
        self.nlp_model = pipeline("text2text-generation", model="google/flan-t5-base")
        
        # Database schema information (would be loaded dynamically in production)
        self.db_schema = {
            "sales": ["date", "product_id", "product_name", "category", "quantity", "price", "total", "region"],
            "inventory": ["product_id", "product_name", "quantity", "reorder_level", "supplier_id", "last_updated"],
            "suppliers": ["supplier_id", "name", "contact", "lead_time_days", "reliability_score"],
            "production": ["date", "product_id", "units_produced", "defect_rate", "production_cost"]
        }
    
    def create_sample_database(self):
        """Create a sample SQLite database for demonstration"""
        conn = sqlite3.connect('sample_business_data.db')
        cursor = conn.cursor()
        
        # Create sales table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sales (
            id INTEGER PRIMARY KEY,
            date TEXT,
            product_id INTEGER,
            product_name TEXT,
            category TEXT,
            quantity INTEGER,
            price REAL,
            total REAL,
            region TEXT
        )
        ''')
        
        # Create inventory table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS inventory (
            product_id INTEGER PRIMARY KEY,
            product_name TEXT,
            quantity INTEGER,
            reorder_level INTEGER,
            supplier_id INTEGER,
            last_updated TEXT
        )
        ''')
        
        # Create suppliers table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS suppliers (
            supplier_id INTEGER PRIMARY KEY,
            name TEXT,
            contact TEXT,
            lead_time_days INTEGER,
            reliability_score REAL
        )
        ''')
        
        # Create production table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS production (
            id INTEGER PRIMARY KEY,
            date TEXT,
            product_id INTEGER,
            units_produced INTEGER,
            defect_rate REAL,
            production_cost REAL
        )
        ''')
        
        # Check if data already exists to avoid duplicates
        cursor.execute("SELECT COUNT(*) FROM sales")
        sales_count = cursor.fetchone()[0]
        
        # Only insert sample data if tables are empty
        if sales_count == 0:
            # Insert sample data
            # Sales data
            products = ["Widget A", "Widget B", "Widget C", "Component X", "Component Y"]
            categories = ["Finished Goods", "Finished Goods", "Finished Goods", "Components", "Components"]
            regions = ["North", "South", "East", "West", "Central"]
            
            for i in range(100):
                product_idx = i % len(products)
                region_idx = i % len(regions)
                date = f"2023-{(i%12)+1:02d}-{(i%28)+1:02d}"
                quantity = np.random.randint(5, 100)
                price = round(np.random.uniform(10, 200), 2)
                total = quantity * price
                
                cursor.execute('''
                INSERT INTO sales (date, product_id, product_name, category, quantity, price, total, region)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (date, product_idx+1, products[product_idx], categories[product_idx], quantity, price, total, regions[region_idx]))
            
            # Inventory data (Avoid duplicate product_id issues)
            for i in range(len(products)):
                cursor.execute('''
                INSERT OR IGNORE INTO inventory (product_id, product_name, quantity, reorder_level, supplier_id, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (i+1, products[i], np.random.randint(50, 500), np.random.randint(20, 100), (i%3)+1, "2023-01-15"))
            
            # Suppliers data
            suppliers = ["Acme Supplies", "BestCo Manufacturing", "Quality Parts Inc"]
            for i in range(len(suppliers)):
                cursor.execute('''
                INSERT OR IGNORE INTO suppliers (supplier_id, name, contact, lead_time_days, reliability_score)
                VALUES (?, ?, ?, ?, ?)
                ''', (i+1, suppliers[i], f"contact{i+1}@example.com", np.random.randint(2, 10), round(np.random.uniform(0.7, 0.99), 2)))
            
            # Production data
            for i in range(50):
                product_idx = i % len(products)
                date = f"2023-{(i%12)+1:02d}-{(i%28)+1:02d}"
                units = np.random.randint(100, 1000)
                defect_rate = round(np.random.uniform(0.01, 0.05), 3)
                cost = round(np.random.uniform(500, 5000), 2)
                
                cursor.execute('''
                INSERT INTO production (date, product_id, units_produced, defect_rate, production_cost)
                VALUES (?, ?, ?, ?, ?)
                ''', (date, product_idx+1, units, defect_rate, cost))
        
        conn.commit()
        conn.close()
    
    def record_audio(self):
        """Record audio from microphone and convert to text"""
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.write("Listening...")
            audio = recognizer.listen(source)
            
        try:
            text = recognizer.recognize_google(audio)
            st.write(f"Recognized: {text}")
            return text
        except sr.UnknownValueError:
            st.error("Could not understand audio")
            return None
        except sr.RequestError:
            st.error("Could not request results from Speech Recognition service")
            return None
    
    def text_to_sql(self, text_query):
        """Convert natural language query to SQL"""
        # Prepare context with schema information
        schema_context = "Database tables and columns:\n"
        for table, columns in self.db_schema.items():
            schema_context += f"- {table}: {', '.join(columns)}\n"
        
        # Create prompt for the model
        prompt = f"{schema_context}\n\nConvert this query to SQL: {text_query}\n\nSQL:"
        
        # Generate SQL query
        try:
            result = self.nlp_model(prompt, max_length=512, do_sample=False)
            sql_query = result[0]['generated_text'].strip()
            
            # Basic validation for safe SQL
            if any(keyword in sql_query.lower() for keyword in ['drop', 'delete', 'update', 'insert', 'alter', 'truncate']):
                return None, "Generated SQL contains potentially harmful operations."
            
            return sql_query, None
        except Exception as e:
            return None, f"Error generating SQL: {str(e)}"
    
    def mock_text_to_sql(self, text_query):
        """Mock function to convert text to SQL without relying on ML model"""
        # This function implements rule-based conversion for common queries
        text_query = text_query.lower()
        
        # Pattern matching for common queries
        if "total sales" in text_query and "region" in text_query:
            return "SELECT region, SUM(total) as total_sales FROM sales GROUP BY region ORDER BY total_sales DESC", None
        
        elif "total sales" in text_query and "product" in text_query:
            return "SELECT product_name, SUM(total) as total_sales FROM sales GROUP BY product_name ORDER BY total_sales DESC", None
        
        elif "sales trend" in text_query or "monthly sales" in text_query:
            return "SELECT strftime('%Y-%m', date) as month, SUM(total) as monthly_sales FROM sales GROUP BY month ORDER BY month", None
        
        elif "inventory" in text_query and "below" in text_query and "reorder" in text_query:
            return "SELECT * FROM inventory WHERE quantity < reorder_level", None
        
        elif "supplier" in text_query and "reliability" in text_query:
            return "SELECT * FROM suppliers ORDER BY reliability_score DESC", None
        
        elif "defect rate" in text_query:
            return "SELECT product_id, AVG(defect_rate) as avg_defect_rate FROM production GROUP BY product_id ORDER BY avg_defect_rate DESC", None
        
        elif "inventory" in text_query and "level" in text_query:
            return "SELECT product_name, quantity, reorder_level FROM inventory", None
        
        # Default query if no pattern is matched
        return "SELECT * FROM sales LIMIT 10", None
    
    def execute_query(self, sql_query):
        """Execute SQL query and return results as DataFrame"""
        try:
            engine = create_engine(self.db_connection_string)
            df = pd.read_sql(sql_query, engine)
            return df, None
        except Exception as e:
            return None, f"Error executing query: {str(e)}"
    
    def visualize_data(self, df):
        """Generate appropriate visualizations based on data characteristics"""
        if df is None or df.empty:
            st.error("No data to visualize")
            return
        
        # Determine the best visualization based on data shape and types
        num_rows, num_cols = df.shape
        
        # Store visualizations
        visualizations = []
        
        # Case 1: Data has 2 columns, one numerical and one categorical
        if num_cols == 2 and any(df.dtypes.apply(lambda x: pd.api.types.is_numeric_dtype(x))):
            num_col = df.select_dtypes(include=['number']).columns[0]
            cat_col = [col for col in df.columns if col != num_col][0]
            
            # Bar chart
            fig = px.bar(df, x=cat_col, y=num_col, title=f"{num_col} by {cat_col}")
            visualizations.append(("Bar Chart", fig))
            
            # Pie chart (if fewer than 10 categories)
            if df[cat_col].nunique() < 10:
                fig = px.pie(df, names=cat_col, values=num_col, title=f"Distribution of {num_col}")
                visualizations.append(("Pie Chart", fig))
        
        # Case 2: Time series data (date/time column detected)
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower() or 'month' in col.lower()]
        if date_cols and any(df.dtypes.apply(lambda x: pd.api.types.is_numeric_dtype(x))):
            date_col = date_cols[0]
            num_cols = df.select_dtypes(include=['number']).columns
            
            for num_col in num_cols[:2]:  # Limit to first 2 numerical columns
                fig = px.line(df, x=date_col, y=num_col, title=f"{num_col} over {date_col}")
                visualizations.append((f"Time Series - {num_col}", fig))
        
        # Case 3: Scatter plot for relationships between numerical variables
        num_cols = df.select_dtypes(include=['number']).columns
        if len(num_cols) >= 2:
            fig = px.scatter(df, x=num_cols[0], y=num_cols[1], 
                           title=f"Relationship between {num_cols[0]} and {num_cols[1]}")
            visualizations.append(("Scatter Plot", fig))
        
        # Case 4: Table view for complex data
        if num_cols > 3 or num_rows > 10:
            fig = go.Figure(data=[go.Table(
                header=dict(values=list(df.columns),
                            fill_color='paleturquoise',
                            align='left'),
                cells=dict(values=[df[col] for col in df.columns],
                           fill_color='lavender',
                           align='left'))
            ])
            fig.update_layout(title="Data Table View")
            visualizations.append(("Table View", fig))
        
        return visualizations

# Streamlit UI for the application
def main():
    st.title("Voice to Visualization System")
    st.write("Query your business data using voice commands and see instant visualizations")
    
    # Initialize the system
    with st.spinner("Initializing system..."):
        system = VoiceToVisualization()
    
    # Interface options
    st.subheader("Query Options")
    query_method = st.radio("Choose input method:", ["Text Input", "Voice Input", "Sample Queries"])
    
    query_text = None
    
    if query_method == "Text Input":
        query_text = st.text_input("Enter your query:", "Show me total sales by region")
        execute = st.button("Execute Query")
    
    elif query_method == "Voice Input":
        if st.button("Record Voice Query"):
            query_text = system.record_audio()
            execute = True
        else:
            execute = False
    
    else:  # Sample Queries
        sample_queries = [
            "Show me total sales by region",
            "What are the monthly sales trends?",
            "Which products have the highest sales?",
            "Show me inventory items below reorder level",
            "What is the defect rate by product?",
            "Show me supplier reliability scores"
        ]
        selected_query = st.selectbox("Select a sample query:", sample_queries)
        query_text = selected_query
        execute = st.button("Execute Query")
    
    # Process the query
    if query_text and execute:
        st.subheader(f"Processing Query: '{query_text}'")
        
        # Convert to SQL
        with st.spinner("Converting to SQL..."):
            # Use mock function for demo purposes
            sql_query, error = system.mock_text_to_sql(query_text)
            if error:
                st.error(error)
            else:
                st.code(sql_query, language="sql")
        
        # Execute query
        if sql_query:
            with st.spinner("Executing query..."):
                df, error = system.execute_query(sql_query)
                if error:
                    st.error(error)
                elif df is not None:
                    st.write("Query Results:")
                    st.dataframe(df)
        
                    # Generate visualizations
                    with st.spinner("Generating visualizations..."):
                        visualizations = system.visualize_data(df)
                        if visualizations:
                            st.subheader("Visualizations")
                            for title, fig in visualizations:
                                st.write(f"**{title}**")
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No suitable visualizations for this data")

if __name__ == "__main__":
    main()