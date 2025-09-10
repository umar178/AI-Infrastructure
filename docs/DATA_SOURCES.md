# 1. Data Sources 

## Original Datasets
1. **Education**  
- **Source:** Pakistan Education Performance Dataset
- **Downloaded From:** [Kaggle Link][https://www.kaggle.com/datasets/mesumraza/pakistan-education-performance-dataset]
- **Temporal Coverage:** 2013 - 2016
- **Key fields:** Boys Enrolled , Complete Primary Schools , Girls Enrolled , Complete Primary Schools , Primary Schools with single classroom Primary Schools with single teacher , All Four Facilities , Any One Facility , Any Three Facilities , Any Two Facilities , Analysis Level Selector , Area (km²) , Bomb Blasts Occurred , Boundary wall , Boundary wall, Building condition satisfactory, Drinking water and 2 more (clusters) , Building condition satisfactory , City , Color By Measure Name , Color By Measure Value , Complete Primary Schools , Country , Drinking water , Drone attacks in Pakistan , Education score , Educational Budget Spend of GDP , Electricity , Enrolment score , Gender parity score , Global Terrorism Index - Pakistan , Learning score , MeasureGroup 1 Measures , MeasureGroup 2 Measures , No Facility , Number of Records , Number of primary schools , Number of primary schools as % of total schools , Number of secondary schools , Number of secondary schools as % of total schools , Other Factors Measure Value , Pakistan Economic Growth , Population , Primary Schools with single classroom , Primary Schools with single teacher , Province , Retention score , School infrastructure score , Show Sheet , Table of Contents , Terrorist Attacks Affectees , Toilet , Total number of schools , Year

2. **Health:**
- **Source:** Pakistan Hospitals Dataset
- **Downloaded From:** [Kaggle Link][https://www.kaggle.com/datasets/mabdullahsajid/pakistan-hopitals-dataset]
- **Key Fields:** Hospital , City , Area , Address , Doctors , Contact

3. **Population:**
- **Source:** Pakistan Census 2017 - Granular Data Area Level
- **Downloaded From:** [Kaggle Link][https://www.kaggle.com/datasets/mesumraza/pakistan-census-2017-granular-data-area-level]
- **Temporal Coverage:** 2017
- **Key Fields:** report_date, province, district, sublvl_01, sublvl_02, sublvl_03, sublvl_04, census_block, population

## 2. Data Integration Strategy

The most significant technical challenge of this project was reconciling the mismatched geographical scales and naming conventions across the three source datasets. Our solution evolved from a planned manual effort to an efficient, data-driven process upon the discovery of a key external resource.

### 2.1. The Core Challenge

The **Population Census** dataset was structured at the `District` level, the official administrative unit for national reporting. In contrast, the **Health** and **Education** datasets were recorded at the `City`/`Town` level, which is often a subdivision of a district. A direct merge was impossible without knowing the parent district for each city. For example:

- A hospital listed in **Khanpur** needed to be correctly mapped to **Haripur District**.
- Schools reported in **Taxila** needed to be aggregated under **Rawalpindi District**.

### 2.2 Solution: Geographical Enrichment via a Master Lookup

**Initial Plan:**
Our first approach was to manually define the administrative hierarchy (Province -> Division -> District) by extracting a unique list from the Census data. We began compiling a crosswalk file (`pakistan_districts_divisions.csv`) to map common city names to their districts.

**Pivot to an Optimized Solution:**
During research, we discovered a far more comprehensive dataset: **["Pakistani Cities, Towns and Villages with District"](https://www.kaggle.com/datasets/imroze/pakistani-cities-towns-and-villages-with-district)** on Kaggle. This dataset provided a pre-built, granular lookup table with over 967 unique locations, including the fields `Name` (City/Town), `Area_Type` (e.g., Village, Town, City), and the critical `District_City` (the parent district).

We adopted this dataset as our master lookup table for several key reasons:

1.  **Comprehensiveness:** It covered a vast number of towns and villages, far exceeding what we could manually compile.
2.  **Accuracy:** It provided official area classifications, removing guesswork and potential errors from manual mapping.
3.  **Efficiency:** It saved significant time and allowed us to focus on analysis rather than data entry.

**Implementation of Enrichment:**

1.  **Cleaning:** City names from all datasets were standardized (lowercased, whitespace stripped, punctuation removed) to ensure robust matching during the join operation.
2.  **Enrichment:** The Health and Education datasets were left-joined with the master lookup table on the cleaned city name. This operation appended the crucial `District`, `Province`, and `Division` columns to each hospital and school record.
3.  **Validation:** We analyzed the merge results for unmatched records and manually resolved a small number of edge cases to ensure maximum data retention.

| Original City (from Source Data) | Resolution from Master Lookup | Final District for Merge |
| :------------------------------- | :---------------------------- | :----------------------- |
| Khanpur                          | Town -> District: Haripur     | Haripur                  |
| Taxila                           | Tehsil -> District: Rawalpindi| Rawalpindi               |
| Lahore City                      | City -> District: Lahore      | Lahore                   |

**Output:** Enriched Health and Education datasets, where each record is tagged with its official `Province`, `Division`, and `District`, ready for consistent aggregation and merging.

## 3. Data Quality Challenges & Solutions

### 3.1 Challenge: Unmatched Geographic Records

Despite using a comprehensive master lookup table, a subset of records (212 schools, 430 hospitals) could not be automatically matched to a district.

**Root cause:**

The mismatch was caused by inconsistencies in naming conventions between the source datasets and the master lookup table:

- **Spelling Variations:** (e.g., Nankana-Sahib vs. Nankana Sahib)
- **Abbreviations:** (e.g., Rwp for Rawalpindi)
- **Alternate Names:** (e.g., Lyallpur for Faisalabad)
- **Newer Settlements:** not yet included in the master lookup dataset.

**Solution:** 
The automated geo-enrichment process successfully matched the vast majority of records. To ensure transparency and facilitate further improvement, all unmatched records were exported to dedicated files for manual review and future mapping.
`unmatched_hospitals.csv:` Contains hospital records requiring district mapping. 
`unmatched_schools.csv:` Contains school records requiring district mapping.
These files serve as a direct input for the next phase of data quality improvement.

### 3.2 Challenge: Correct Aggregation of Percentage Data
A critical methodological challenge was correctly handling the enrollment data from the Education dataset, which was provided as percentages (% Boys Enrolled, % Girls Enrolled).

**Problem:**
The initial approach incorrectly applied a .sum() aggregation to these percentage fields. This is a fundamental error in data analysis, as summing percentages across different entities (schools) produces a meaningless and invalid result. For example, summing 60% from one school and 70% from another does not equal 130%; it has no mathematical or practical interpretation.

**Solution:**
To derive accurate and meaningful insights, the methodology was corrected to:

- **1. Clean and Convert:** The percentage strings (e.g., "62.50%") were converted to numeric values (e.g., 62.5).

- **2. Calculate Averages:** The correct aggregation for percentages is the mean. We calculated the avg_boys_enrolled_pct and avg_girls_enrolled_pct for each district.

- **3. Maintain Integrity:** This approach truthfully represents the data as it was provided—showing the average gender distribution across schools in a district—without inventing underlying numbers that were not available.

### Resulting Insight:
The final dataset provides a powerful and truthful metric: the average percentage of boys and girls enrolled in schools for each district. This allows for robust analysis of gender parity in education across Pakistan without compromising data integrity.