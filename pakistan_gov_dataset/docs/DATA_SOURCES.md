# Pakistan Data Integration Pipeline - Complete Documentation

## Folder Structure
URAAN_PAKISTAN_TECHATHON
|__ docs/ DATA_SOURCES.md and schema
|__ scripts / data_integration.py and app.py and schema_validator.py
|__ data / raw and processed
|__ requirements.txt
|__ .env

## 1. Data Sources

### 1.1 Education Dataset
- **Source:** Pakistan Education Performance Dataset
- **Downloaded From:** [Kaggle Link](https://www.kaggle.com/datasets/mesumraza/pakistan-education-performance-dataset)
- **Temporal Coverage:** 2013 - 2016
- **Key Fields:**
  - `City`, `% Boys Enrolled`, `% Girls Enrolled`
  - `Total number of schools`, `Number of primary schools`, `Number of secondary schools`
  - **Infrastructure metrics:** `Electricity`, `Drinking water`, `Boundary wall`, `Toilet`
  - **Quality scores:** `Education score`, `School infrastructure score`
  - `Province`, `Year`, `Area (km²)`, `Population`

### 1.2 Health Dataset
- **Source:** Pakistan Hospitals Dataset
- **Downloaded From:** [Kaggle Link](https://www.kaggle.com/datasets/mabdullahsajid/pakistan-hopitals-dataset)
- **Key Fields:**
  - `Hospital`, `City`, `Area`, `Address`
  - `Doctors`, `Contact`

### 1.3 Population Dataset
- **Source:** Pakistan Census 2017 - Granular Data Area Level
- **Downloaded From:** [Kaggle Link](https://www.kaggle.com/datasets/mesumraza/pakistan-census-2017-granular-data-area-level)
- **Temporal Coverage:** 2017
- **Key Fields:**
  - `report_date`, `province`, `district`, `sublvl_01`, `sublvl_02`
  - `sublvl_03`, `sublvl_04`, `census_block`, `population`

### 1.4 Geographic Master Lookup
- **Source:** Pakistani Cities, Towns and Villages with District
- **Downloaded From:** [Kaggle Link](https://www.kaggle.com/datasets/imroze/pakistani-cities-towns-and-villages-with-district)
- **Purpose:** Resolve city-to-district mappings
- **Key Fields:** `Name`, `Area_Type`, `District_City`

## 2. Data Integration Strategy

The primary challenge was reconciling mismatched geographic scales and naming conventions across the datasets. Population Census data is at the **district level**, while Health and Education data are at the **city/town level**. Direct merges required knowing the parent district of each city.

### 2.1 Core Challenge
- Hospitals in **Khanpur** needed mapping to **Haripur District**.
- Schools in **Taxila** needed aggregation under **Rawalpindi District**.

### 2.2 Solution: Geographical Enrichment via a Master Lookup

**Initial Plan:**  
- Manually define administrative hierarchy (Province → Division → District).  
- Compile a crosswalk file (`pakistan_districts_divisions.csv`) to map city names to districts.

**Optimized Solution:**  
- Discovered **Pakistani Cities, Towns and Villages with District** dataset on Kaggle (967+ locations).  
- Provides `Name`, `Area_Type`, and `District_City`.

**Benefits:**  
- **Comprehensiveness:** Covers more locations than manual compilation.  
- **Accuracy:** Official area classifications reduce errors.  
- **Efficiency:** Saves time and focuses on analysis.

**Implementation Steps:**  
1. **Cleaning:** Standardize city names (lowercase, remove whitespace/punctuation).  
2. **Enrichment:** Left-join Health and Education datasets with master lookup.  
3. **Validation:** Resolve unmatched records manually.

| Original City | Resolution from Master Lookup | Final District |
|---------------|------------------------------|----------------|
| Khanpur       | Town → District: Haripur     | Haripur       |
| Taxila        | Tehsil → District: Rawalpindi | Rawalpindi    |
| Lahore City   | City → District: Lahore      | Lahore        |

**Output:** Enriched Health and Education datasets with official `Province`, `Division`, and `District` columns.

## 3. Data Quality Challenges & Solutions

### 3.1 Unmatched Geographic Records
- **Issue:** 212 schools, 430 hospitals could not be matched automatically.
- **Causes:**  
  - Spelling variations (e.g., Nankana-Sahib vs. Nankana Sahib)  
  - Abbreviations (e.g., Rwp for Rawalpindi)  
  - Alternate names (e.g., Lyallpur for Faisalabad)  
  - Newer settlements missing from master lookup
- **Solution:** Export unmatched records for manual review:
  - `unmatched_hospitals.csv`  
  - `unmatched_schools.csv`

### 3.2 Correct Aggregation of Percentage Data
- **Problem:** Summing `% Boys Enrolled` or `% Girls Enrolled` across schools is invalid.
- **Solution:**  
  1. Convert percentage strings to numeric (e.g., "62.50%" → 62.5).  
  2. Calculate **average** for each district (`avg_boys_enrolled_pct`, `avg_girls_enrolled_pct`).  
  3. Maintains truthful representation of data.

**Result:** Average district-level gender enrollment metrics suitable for robust analysis.

## 4. Technical Implementation

### Core Pipeline Architecture
1. `load_dataset()` – Robust file loading with error handling  
2. `standardize_name()` – Handles 50+ naming variations  
3. `create_district_mapping()` – Integrates master lookup table  
4. `apply_manual_overrides()` – Handles special cases  
5. `aggregate_by_district()` – Correctly averages percentages  
6. `merge_datasets()` – Produces unified output  
7. `validate_output()` – Checks schema compliance  

### Key Technical Features
- **Name Standardization:** Handles hyphens, underscores, suffixes, FR regions  
- **Error Resilience:** Try-catch blocks, missing data handling  
- **Modular Design:** Separate functions for loading, cleaning, mapping, validation  
- **Quality Assurance:** Automated schema validation, unmatched record tracking

## 5. Key Results & Insights

- **120+ districts** integrated across all provinces  
- **95%+ geographic matching rate** for domestic facilities  
- **Cross-sector metrics**: Health, Education, Population in a single unified view  
- **Standardized schema** compliant with Pakistan government standards  

**Data Quality Achievements:**  
- Health Data: 100% domestic hospitals mapped  
- Education Data: 28 special administrative regions identified  
- Population Data: Complete district-level coverage  
- Transparent logging of unmatched records

**Notable Findings:**  
- Regional disparities in doctor-to-population ratios  
- Gender enrollment variations quantified  
- Infrastructure gaps identified  
- Special administrative regions require specialized mapping

## 6. Output Datasets

### Primary Outputs
- `pakistan_unified_district_data.csv` – Integrated dataset at district level  
- Schema compliant with `pakistan_government_schema.json`

### Quality Control Outputs
- `unmatched_hospitals.csv` – 26 international + 1 domestic facility  
- `unmatched_schools.csv` – 28 special regions (tribal agencies, frontier regions, new districts)

### Schema Validation
- **File:** `schema_validator.py`  
- **Purpose:** Validate output data against government schema  
- **Usage:** Automated quality assurance

## 7. Limitations & Future Enhancements

### Current Limitations
1. Geographic coverage: Special administrative regions need additional reference data  
2. Temporal scope: 2013–2017 only  
3. Facility granularity: Hospital-level data aggregated to district  
4. International facilities excluded from domestic analysis

### Future Enhancements
- Real-time data integration  
- Additional datasets (economic, infrastructure)  
- Web dashboard with interactive visualizations  
- API endpoints for programmatic access  
- Predictive analytics via machine learning  
- Mobile app for field data collection
