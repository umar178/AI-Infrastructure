"""
JSON Schema Validator for Pakistan Government Data
Validates datasets against the unified schema
"""

import json
import jsonschema
from jsonschema import validate
import pandas as pd
import os

class GovernmentDataValidator:
    def __init__(self, schema_path):
        """Initialize validator with schema"""
        with open(schema_path, 'r') as f:
            self.schema = json.load(f)
    
    def validate_json_data(self, json_data):
        """Validate JSON data against schema"""
        try:
            validate(instance=json_data, schema=self.schema)
            return {
                "is_valid": True,
                "errors": []
            }
        except jsonschema.ValidationError as e:
            return {
                "is_valid": False,
                "errors": [str(e)]
            }
    
    def validate_dataframe(self, df, data_type="unified"):
        """Validate pandas DataFrame against schema components"""
        errors = []
        
        # Check required geographic fields
        if 'province' not in df.columns or 'district' not in df.columns:
            errors.append("Missing required geographic fields: province, district")
        
        # Check data types and constraints based on data_type
        if data_type == "unified":
            errors.extend(self._validate_unified_data(df))
        elif data_type == "health":
            errors.extend(self._validate_health_data(df))
        elif data_type == "education":
            errors.extend(self._validate_education_data(df))
        elif data_type == "population":
            errors.extend(self._validate_population_data(df))
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "records_checked": len(df)
        }
    
    def _validate_unified_data(self, df):
        """Validate unified district dataset"""
        errors = []
        
        # Check required fields
        required_fields = ['province', 'district', 'total_population']
        for field in required_fields:
            if field not in df.columns:
                errors.append(f"Missing required field: {field}")
        
        # Check data types and constraints
        if 'total_population' in df.columns:
            if (df['total_population'] < 0).any():
                errors.append("total_population contains negative values")
        
        if 'avg_boys_enrolled_pct' in df.columns:
            if ((df['avg_boys_enrolled_pct'] < 0) | (df['avg_boys_enrolled_pct'] > 100)).any():
                errors.append("avg_boys_enrolled_pct outside valid range (0-100)")
        
        if 'avg_girls_enrolled_pct' in df.columns:
            if ((df['avg_girls_enrolled_pct'] < 0) | (df['avg_girls_enrolled_pct'] > 100)).any():
                errors.append("avg_girls_enrolled_pct outside valid range (0-100)")
        
        return errors
    
    def _validate_health_data(self, df):
        """Validate health facility data"""
        errors = []
        if 'DOCTORS' in df.columns and (df['DOCTORS'] < 0).any():
            errors.append("DOCTORS contains negative values")
        return errors
    
    def _validate_education_data(self, df):
        """Validate education data"""
        errors = []
        percentage_columns = ['% Boys Enrolled', '% Girls Enrolled']
        for col in percentage_columns:
            if col in df.columns:
                # Convert percentage strings to numbers for validation
                try:
                    percentages = df[col].str.rstrip('%').astype(float)
                    if ((percentages < 0) | (percentages > 100)).any():
                        errors.append(f"{col} outside valid range (0-100%)")
                except:
                    errors.append(f"{col} contains invalid percentage values")
        return errors
    
    def generate_validation_report(self, datasets):
        """Generate comprehensive validation report"""
        report = {
            "schema_version": self.schema.get("version", "1.0.0"),
            "validation_date": pd.Timestamp.now().isoformat(),
            "datasets_validated": [],
            "summary": {
                "total_datasets": 0,
                "valid_datasets": 0,
                "total_errors": 0
            }
        }
        
        for name, (df, data_type) in datasets.items():
            result = self.validate_dataframe(df, data_type)
            dataset_report = {
                "dataset_name": name,
                "data_type": data_type,
                "is_valid": result["is_valid"],
                "records_checked": result["records_checked"],
                "errors": result["errors"]
            }
            report["datasets_validated"].append(dataset_report)
            
            report["summary"]["total_datasets"] += 1
            if result["is_valid"]:
                report["summary"]["valid_datasets"] += 1
            report["summary"]["total_errors"] += len(result["errors"])
        
        return report

def main():
    """Example usage of the validator"""
    validator = GovernmentDataValidator("docs/pakistan_government_schema.json")
    try:
        unified_df = pd.read_csv("data/processed/pakistan_unified_district_data.csv")
        result = validator.validate_dataframe(unified_df, "unified")
        
        print("=== VALIDATION REPORT ===")
        print(f"Valid: {result['is_valid']}")
        print(f"Records checked: {result['records_checked']}")
        if not result['is_valid']:
            print("Errors:")
            for error in result['errors']:
                print(f"  - {error}")
        else:
            print("All data validates against schema!")
            
    except Exception as e:
        print(f"Validation failed: {e}")

if __name__ == "__main__":
    main()