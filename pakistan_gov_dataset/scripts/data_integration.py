"""
Complete pipeline to Integrate Pakistan Health, Education, and Population Data.
Uses a master lookup table to resolve city-to-district mapping.
"""

import pandas as pd
import os


DATA_PATH = "data/raw/"
OUTPUT_PATH = "data/processed/"

def standardize_name(name):
    """
    Cleans and standardizes text for better matching.
    Converts to lowercase, removes extra spaces/special characters.
    """
    try:
        # Handle missing values
        if pd.isna(name):
            return name
        
        # Convert to string and basic cleaning
        name = str(name).lower().strip()

        # Convert hypens to spaces:
        name = name.replace('-' , ' ')
        name = name.replace('_' , ' ')
        
        # Remove 'district' if present at the end
        if name.endswith('district'):
            name = name[:-8].strip()  

        # Handle FR (Frontier Region) names
        if name.startswith('fr'):
            name = name[2:].strip()

        # Remove agency suffix if present
        if name.endswith('agency'):
            name = name[:-7].strip()
        
        # Keep only letters, numbers
        name = ''.join(char for char in name if char.isalnum())
        
        return name
        
    except Exception as e:
        print(f"Error cleaning name '{name}': {e}")
        return name  # Return original if error occurs

def load_dataset(file_path, dataset_name):
    """Safely load a CSV file with error handling."""
    try:
        if not os.path.exists(file_path):
            print(f"Error: File not found - {file_path}")
            return None
            
        df = pd.read_csv(file_path)
        print(f"Loaded {dataset_name}: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
        
    except Exception as e:
        print(f"Error loading {dataset_name} from {file_path}: {e}")
        return None

def main():
    """Main function to run the data integration pipeline."""
    print("Starting Pakistan Data Integration Pipeline...")
    print("=" * 50)
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # 1. Load all datasets
    print("\n1. Loading datasets...")
    health_df = load_dataset(DATA_PATH + "health_dataset.csv", "Health Data")
    education_df = load_dataset(DATA_PATH + "education_dataset.csv", "Education Data") 
    population_df = load_dataset(DATA_PATH + "population_dataset.csv", "Population Data")
    lookup_df = load_dataset(DATA_PATH + "pakistan_area_hierarchy.csv", "Lookup Data")
    
    # Check if any dataset failed to load
    if any(df is None for df in [health_df, education_df, population_df, lookup_df]):
        print("Error: Could not load one or more datasets. Please check file paths.")
        return
    
    # 2. Clean location names for better matching
    print("2. Cleaning location names...")
    try:
        health_df["city_clean"] = health_df["CITY"].apply(standardize_name)
        education_df["city_clean"] = education_df["City"].apply(standardize_name)
        population_df["district_clean"] = population_df["district"].apply(standardize_name)
        
        lookup_df["name_clean"] = lookup_df["Name"].apply(standardize_name)
        lookup_df["district_city_clean"] = lookup_df["District_City"].apply(standardize_name)
        
        print("All location names standardized")
        
    except KeyError as e:
        print(f"Error: Missing expected column {e}")
        return
    except Exception as e:
        print(f"Error during name standardization: {e}")
        return

    # 3. Add district information to health and education data using two-step merge
    print("3. Adding district information with two-step merge...")
    try:
        # Create district mapping from both Name and District_City columns
        district_mapping = {}
        for _, row in lookup_df.iterrows():
            if not pd.isna(row['name_clean']):
                district_mapping[row['name_clean']] = row['District_City']
            if not pd.isna(row['district_city_clean']):
                district_mapping[row['district_city_clean']] = row['District_City']

        # Add manual mappings for specific cases
        manual_mappings = {
            'bajauragency': 'Bajaur', 'bhimber': 'Bhimber', 'shahdadpur': 'Sanghar',
            'shahkot': 'Nankana Sahib', 'talagang': 'Chakwal', 'tankcity': 'Tank',
            'tarbela': 'Haripur', 'umarkot': 'Umarkot', 'wahcantt': 'Rawalpindi',
            'mithi': 'Tharparkar', 'kashmor': 'Kashmore', 'dunyaput': 'Lodhran',
            'dunyapur': 'Lodhran', 'kamoke': 'Gujranwala', 'alipur': 'Muzaffargarh',
            'kandiaro': 'Naushahro Feroze', 'buner': 'Buner', 'baden': 'Swat',
            'dargai': 'Malakand', 'astor': 'Astore', 'kashmorekandhkot': 'Kashmore',
            'naushehroferoze': 'Naushahro Firoz', 'kohistan': 'Upper Kohistan',
            'ghanci': 'Ghizer', 'ghanchi': 'Ghizer',
            
            # Education data specific mappings
            'hattian': 'Hattian', 'haveli': 'Haveli', 'neelum': 'Neelum',
            'sudhnutti': 'Sudhnati', 'torgar': 'Tor Ghar', 'diamir': 'Diamer',
            'hunzanagar': 'Hunza', 'musakhail': 'Musakhel', 'killasaifullah': 'Qila Saifullah',
            'kambarshahdadkot': 'Shahdad Kot', 'harnai': 'Harnai', 'frdikhan': 'Dera Ismail Khan',
            'batagram': 'Battagram', 'jhalmagsi': 'Jhal Magsi', 'kachhi': 'Kachhi',
            'nushki': 'Nushki', 'sherani': 'Sherani', 'washuk': 'Washuk',
        }
        district_mapping.update(manual_mappings)
        # Add district column to health data
        health_enriched = health_df.copy()
        health_enriched['District_City'] = health_enriched['city_clean'].map(district_mapping)
        
        # THEN manually fix only the specific problem cases
        health_enriched.loc[health_enriched['city_clean'] == 'bajauragency', 'District_City'] = 'Bajaur'
        health_enriched.loc[health_enriched['city_clean'] == 'bhimber', 'District_City'] = 'Bhimber'

        # Add district column to education data
        education_enriched = education_df.copy()
        education_enriched['District_City'] = education_enriched['city_clean'].map(district_mapping)
        
        print("District information added to health and education data")
        print(f"   - Health records with district: {health_enriched['District_City'].notna().sum()}/{len(health_enriched)}")
        print(f"   - Education records with district: {education_enriched['District_City'].notna().sum()}/{len(education_enriched)}")
        
    except Exception as e:
        print(f"Error during data enrichment: {e}")
        return
    
    # 4. Check for unmatched records
    print("4. Checking data quality...")
    health_unmatched = health_enriched[health_enriched['District_City'].isna()]
    education_unmatched = education_enriched[education_enriched['District_City'].isna()]

    print(f"   - Hospitals without district match: {len(health_unmatched)}")
    print(f"   - Schools without district match: {len(education_unmatched)}")

    # Save unmatched records for review - ONLY HOSPITAL NAME AND CITY
    if len(health_unmatched) > 0:
        health_unmatched[['HOSPITAL NAME', 'CITY']].to_csv(OUTPUT_PATH + 'unmatched_hospitals.csv', index=False)
        print("   - Saved unmatched hospitals to 'unmatched_hospitals.csv'")

    if len(education_unmatched) > 0:
        education_unmatched[['City', '% Boys Enrolled']].to_csv(OUTPUT_PATH + 'unmatched_schools.csv', index=False)
        print("   - Saved unmatched schools to 'unmatched_schools.csv'")

    # 5. Aggregate data to district level
    print("5. Aggregating data by district...")
    try:
        # Aggregate population data
        population_agg = population_df.groupby(['province', 'district_clean']).agg(
            total_population=('population', 'sum')
        ).reset_index()

        # Aggregate health data
        health_agg = health_enriched.groupby(['District_City']).agg(
            number_of_hospitals=('HOSPITAL NAME', 'count'),
            total_doctors=('DOCTORS', 'sum')
        ).reset_index()
        
        # Clean doctors data
        health_agg['total_doctors'] = pd.to_numeric(health_agg['total_doctors'], errors='coerce').fillna(0).astype(int)
        health_agg['district_clean'] = health_agg['District_City'].apply(standardize_name)

        # Aggregate education data
        # First, clean the percentage columns and convert them to numbers
        def convert_percentage(value):
            """
            Safely converts a percentage string (e.g., "62.50%") to a float (e.g., 62.5).
            """
            try:
                if pd.isna(value):
                    return None
                if isinstance(value, (int, float)):
                    return float(value)
                if isinstance(value, str):
                    clean_value = value.replace('%', '').strip()
                    return float(clean_value)
                return None
            except Exception as e:
                print(f"Warning: Could not convert value '{value}'. Error: {e}")
                return None

        education_enriched['boys_pct_clean'] = education_enriched['% Boys Enrolled'].apply(convert_percentage)
        education_enriched['girls_pct_clean'] = education_enriched['% Girls Enrolled'].apply(convert_percentage)

        education_agg = education_enriched.groupby(['District_City']).agg(
            number_of_schools=('Total number of schools', 'sum'),
            avg_boys_enrolled_pct=('boys_pct_clean', 'mean'),   
            avg_girls_enrolled_pct=('girls_pct_clean', 'mean')  
        ).reset_index()
        education_agg['district_clean'] = education_agg['District_City'].apply(standardize_name)
        
        print("Data aggregation completed")
        
    except Exception as e:
        print(f"Error during data aggregation: {e}")
        return

    # 6. Merge all datasets together
    print("6. Merging all datasets...")
    try:
        # Merge population with health data
        unified_dataset = population_agg.merge(
            health_agg,
            how='left',
            on='district_clean'
        )

        # Merge with education data
        unified_dataset = unified_dataset.merge(
            education_agg,
            how='left',
            on='district_clean'
        )

        # Clean up duplicate columns from merging
        columns_to_drop = [col for col in unified_dataset.columns if col.endswith(('_x', '_y'))]
        if columns_to_drop:
            unified_dataset = unified_dataset.drop(columns=columns_to_drop)

        count_columns = ['number_of_hospitals', 'total_doctors', 'number_of_schools']
        pct_columns = ['avg_boys_enrolled_pct', 'avg_girls_enrolled_pct']

        for col in count_columns:
            if col in unified_dataset.columns:
                unified_dataset[col] = unified_dataset[col].fillna(0)

        for col in pct_columns:
            if col in unified_dataset.columns:
                unified_dataset[col] = unified_dataset[col].fillna(0)

        final_columns = [
            'province', 'district_clean', 'total_population', 'number_of_hospitals',
            'total_doctors', 'number_of_schools', 'avg_boys_enrolled_pct', 'avg_girls_enrolled_pct'
        ]
        
        # Keep only columns that exist
        final_columns = [col for col in final_columns if col in unified_dataset.columns]
        unified_dataset = unified_dataset[final_columns]

        # Rename for clarity
        unified_dataset = unified_dataset.rename(columns={'district_clean': 'district'})
        
        print("All datasets merged successfully")
        
    except Exception as e:
        print(f"Error during data merging: {e}")
        return

    # 7. Save the final result
    print("7. Saving results...")
    try:
        output_file = OUTPUT_PATH + 'pakistan_unified_district_data.csv'
        unified_dataset.to_csv(output_file, index=False)
        
        print("Final dataset saved successfully!")
        print(f"   - File: {output_file}")
        print(f"   - Shape: {unified_dataset.shape[0]} districts, {unified_dataset.shape[1]} features")
        
    except Exception as e:
        print(f"Error saving final dataset: {e}")
        return

    # 8. Show final summary
    print("\n" + "=" * 50)
    print("INTEGRATION COMPLETE - SUMMARY")
    print("=" * 50)
    
    print(f"Total districts: {len(unified_dataset)}")
    print(f"Total population: {unified_dataset['total_population'].sum():,}")
    print(f"Total hospitals: {unified_dataset['number_of_hospitals'].sum():,}")
    print(f"Total doctors: {unified_dataset['total_doctors'].sum():,}")
    print(f"Total schools: {unified_dataset['number_of_schools'].sum():,}")

    avg_boys = unified_dataset['avg_boys_enrolled_pct'].mean(skipna=True)
    avg_girls = unified_dataset['avg_girls_enrolled_pct'].mean(skipna=True)
    print(f"National Avg. Boys Enrollment: {avg_boys:.1f}%")
    print(f"National Avg. Girls Enrollment: {avg_girls:.1f}%")
    
    print("First few districts:")
    print(unified_dataset.head(10).to_string(index=False))
    
    print("Pipeline completed successfully!")

    # Add district column to health data
    health_enriched = health_df.copy()
    health_enriched['District_City'] = health_enriched['city_clean'].map(district_mapping)

if __name__ == "__main__":
    main()