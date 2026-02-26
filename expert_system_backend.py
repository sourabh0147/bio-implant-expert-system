import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import os

# --- 1. Preprocessing for Friction (COF) ---
def preprocess_cof_data(df):
    processed_data = []
    
    # Map raw CSV headers (if they differ) to the New Standard Names
    # Adjust the keys on the left if your Friction CSV has different headers
    alloy_map = {
        'Mg': 'Pure Mg', 
        'Mg bi': 'Al-Mg-Bi', 
        'Mg Sr': 'Al-Mg-Sr', 
        'Mg Zn': 'Al-Mg-Zn'
    }

    # Iterate through the map to find columns
    for search_term, standard_name in alloy_map.items():
        # Look for columns containing the search term (e.g., "Mg bi")
        timestamp_col = next((col for col in df.columns if search_term in col and 'Timestamp' in col), None)
        cof_col = next((col for col in df.columns if search_term in col and 'COF' in col), None)

        if timestamp_col and cof_col:
            temp_df = df[[timestamp_col, cof_col]].copy()
            temp_df.columns = ['Timestamp', 'COF']
            temp_df['Alloy_Type'] = standard_name # Assign the correct Al- name
            processed_data.append(temp_df)

    if not processed_data:
        return pd.DataFrame()
    
    concatenated_df = pd.concat(processed_data, ignore_index=True)
    concatenated_df.dropna(subset=['COF'], inplace=True)
    return concatenated_df

# --- 2. Preprocessing for OCP (Corrosion) ---
def preprocess_ocp_data(df):
    processed_data = []
    
    # Map OCP.csv headers to New Standard Names
    # Note the specific spacing in 'Al- Mg-Bi' from your source file
    header_map = {
        'Pure Mg': 'Pure Mg', 
        'Al- Mg-Bi': 'Al-Mg-Bi', 
        'Al-Mg-Sr': 'Al-Mg-Sr', 
        'Al-Mg-Zn': 'Al-Mg-Zn'
    }
    
    for i in range(0, len(df.columns), 2):
        if i+1 >= len(df.columns): break
        col_name = df.columns[i]
        
        # Find match in header map
        clean_name = next((val for key, val in header_map.items() if key in col_name), None)
        
        if clean_name:
            temp_df = df.iloc[:, i:i+2].copy()
            temp_df.columns = ['Timestamp', 'OCP']
            
            # Force numeric conversion
            temp_df['Timestamp'] = pd.to_numeric(temp_df['Timestamp'], errors='coerce')
            temp_df['OCP'] = pd.to_numeric(temp_df['OCP'], errors='coerce')
            
            temp_df['Alloy_Type'] = clean_name
            temp_df.dropna(inplace=True)
            processed_data.append(temp_df)

    if not processed_data:
        return pd.DataFrame()

    return pd.concat(processed_data, ignore_index=True)

# --- 3. Build Wear Database from Excel Sheets ---
def build_wear_database(file_path="Wear proflie.xlsx"):
    print(f"Processing Wear Database from: {file_path}")
    wear_db = {}
    
    # Map Sheet Names to New Standard Names
    sheet_map = {
        'Pure Mg': 'Pure Mg', 'PureMg': 'Pure Mg', 'Mg': 'Pure Mg',
        
        'Al-Mg-Bi': 'Al-Mg-Bi', 'AlMgBi': 'Al-Mg-Bi', 'Mg-Bi': 'Al-Mg-Bi', 'Mg bi': 'Al-Mg-Bi',
        
        'Al-Mg-Sr': 'Al-Mg-Sr', 'AlMgSr': 'Al-Mg-Sr', 'Mg-Sr': 'Al-Mg-Sr', 'Mg Sr': 'Al-Mg-Sr',
        
        'Al-Mg-Zn': 'Al-Mg-Zn', 'AlMgZn': 'Al-Mg-Zn', 'Mg-Zn': 'Al-Mg-Zn', 'Mg Zn': 'Al-Mg-Zn'
    }

    try:
        xls = pd.read_excel(file_path, sheet_name=None)
        
        # Sort keys by length to match specific names before general ones (e.g. AlMgBi before Mg)
        sorted_keys = sorted(sheet_map.keys(), key=len, reverse=True)

        for sheet_name, df in xls.items():
            standard_name = None
            for key in sorted_keys:
                if key.lower() in sheet_name.lower(): 
                    standard_name = sheet_map[key]
                    break 
            
            if standard_name:
                try:
                    z_values = df.iloc[:, 2] 
                    max_depth = z_values.max() - z_values.min()
                    
                    wear_db[standard_name] = {
                        'max_depth_um': float(f"{max_depth:.2f}"),
                        'wear_area_um2': 0 
                    }
                    print(f"  -> [MATCH] '{sheet_name}' mapped to '{standard_name}' -> Depth: {max_depth:.2f} um")
                except Exception as e:
                    print(f"  -> [Error] processing sheet {sheet_name}: {e}")

        if wear_db:
            joblib.dump(wear_db, 'wear_database.pkl')
            print("Wear database saved as 'wear_database.pkl'")

    except Exception as e:
        print(f"Error building wear DB: {e}")

# --- 4. Main Training Pipeline ---
def train_and_save_systems():
    # 1. Train COF
    try:
        print("\n--- Training Friction Model ---")
        df_cof = pd.read_csv("Friction_File.csv", header=[0,1])
        
        # [Insert your column renaming logic here if needed from original file]
        # Minimal renaming for robustness:
        new_columns = []
        for col_pair in df_cof.columns:
            if 'Unnamed' in col_pair[0] or col_pair[0] == 'Mg,':
                prev = new_columns[-1].split('_')[0] if new_columns else 'Unknown'
                suffix = 'COF' if 'COF' in col_pair[1] else col_pair[1].strip()
                new_columns.append(f'{prev}_{suffix}')
            else:
                new_columns.append(f'{col_pair[0].strip()}_{col_pair[1].strip()}')
        df_cof.columns = new_columns

        processed_cof = preprocess_cof_data(df_cof)
        
        if not processed_cof.empty:
            X_cof = processed_cof[['Timestamp', 'Alloy_Type']]
            y_cof = processed_cof['COF']

            preprocessor = ColumnTransformer([
                ('num', StandardScaler(), ['Timestamp']),
                ('cat', OneHotEncoder(handle_unknown='ignore'), ['Alloy_Type'])
            ])
            
            cof_model = Pipeline([('prep', preprocessor), ('rf', RandomForestRegressor(random_state=42))])
            cof_model.fit(X_cof, y_cof)
            joblib.dump(cof_model, 'random_forest_model.pkl')
            print("Friction model saved.")
        else:
            print("Error: No COF data found. Check CSV headers.")

    except Exception as e:
        print(f"Skipping COF: {e}")

    # 2. Train OCP
    try:
        print("\n--- Training Corrosion Model ---")
        df_ocp = pd.read_csv("OCP.csv") 
        processed_ocp = preprocess_ocp_data(df_ocp)
        
        if not processed_ocp.empty:
            X_ocp = processed_ocp[['Timestamp', 'Alloy_Type']]
            y_ocp = processed_ocp['OCP']
            
            preprocessor = ColumnTransformer([
                ('num', StandardScaler(), ['Timestamp']),
                ('cat', OneHotEncoder(handle_unknown='ignore'), ['Alloy_Type'])
            ])
            ocp_model = Pipeline([('prep', preprocessor), ('rf', RandomForestRegressor(random_state=42))])
            ocp_model.fit(X_ocp, y_ocp)
            joblib.dump(ocp_model, 'ocp_model.pkl')
            print("OCP model saved.")
    except Exception as e:
        print(f"Skipping OCP: {e}")

    # 3. Build Wear DB
    print("\n--- Building Wear Database ---")
    build_wear_database("Wear proflie.xlsx")

if __name__ == "__main__":
    train_and_save_systems()