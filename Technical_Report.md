# Phase 1 Project Report – Pakistan National AI Cloud & API Gateway
# Resources
- Github
- Dataset
- Fine tuned models

# Introduction
This document presents the work completed during Phase 1 of the hackathon, focused on building the foundation for the Pakistan National AI Cloud & API Gateway. The aim was to address the lack of unified AI infrastructure in Pakistan by designing common government data schemas, training Urdu NLP models, and creating secure RESTful APIs. This phase lays the groundwork for a commercialization-ready infrastructure to serve government, academia, and startups.

# Problem Solving and Approach

1. Data Schema Design
A major challenge was the absence of standardized formats across government datasets. We designed JSON-based schema templates for three domains: health, education, and population. These schemas were created to ensure reusability, compatibility, and compliance with potential national standards. This work is available in the GitHub repository: https://github.com/umar178/AI-Infrastructure/tree/Trainer

## 	Limitations:
During the integration process, one of the main challenges was reconciling mismatched geographic levels across datasets, health and education data existed at the city level, while population data was available only at the district level. This inconsistency made direct merging difficult. Additionally, several city names did not exactly match the district names in the population dataset due to spelling variations, abbreviations, and legacy names (e.g., “Lyallpur” for Faisalabad).
A few special administrative regions and recently formed districts lacked representation in the master lookup, resulting in unmatched records. Furthermore, the limited temporal coverage (2013–2017) restricted the ability to perform trend analysis.

## 	Solutions:
To overcome these limitations, a Geographic Master Lookup Table was introduced, which mapped each city or town to its corresponding district. This lookup greatly improved merge accuracy and ensured consistent district-level aggregation across all datasets. Manual verification was also conducted for unmatched entries to improve completeness. For future improvements, expanding the master reference file, including updated regional data, and automating name standardization through fuzzy matching or machine learning-based entity resolution are recommended. Extending temporal coverage and integrating real-time data sources through APIs will further strengthen the pipeline’s reliability.

## 	API Impletation Usig FastAPI
The District Data API was implemented using FastAPI, a modern, high-performance Python web framework for building RESTful APIs. FastAPI was chosen for its speed, automatic generation of OpenAPI (Swagger) documentation, and support for asynchronous request handling, which is essential for efficiently serving large datasets.
The API provides multiple endpoints for accessing and querying district-level data:
•	GET / – Root endpoint for a simple health check.
•	GET /data – Returns all district records in JSON format.
•	GET /data/{id} – Retrieves a single record by its index.
•	GET /filter – Filters districts based on a query parameter district.
•	GET /info – Provides metadata about the loaded dataset, including the number of records, available fields, and dataset coverage.
To secure access to sensitive data, the API uses Bearer token authentication. Requests to endpoints that return data require an Authorization header with a valid API key. Additionally, rate limiting was implemented using the SlowAPI library to prevent abuse: /data allows 100 requests per minute, /data/{id} allows 50 requests per minute, and /filter allows 30 requests per minute. Requests exceeding these limits return a 429 Too Many Requests error.
The API can be run locally using Uvicorn, a lightning-fast ASGI server, with hot reload enabled for development. Once the server is running, interactive documentation is automatically available at /docs (Swagger UI), allowing developers to view endpoints, read their descriptions, and test API calls directly from the browser. The OpenAPI specification is also exposed at /openapi.json for integration with other tools.
Sample API usage:
# Health check
curl http://127.0.0.1:8000/docs

# Get all district data
curl -H "Authorization: Bearer <API_KEY>" http://127.0.0.1:8000/data

# Get single record by index
curl -H "Authorization: Bearer <API_KEY>" http://127.0.0.1:8000/data/0

# Filter by district
curl -H "Authorization: Bearer <API_KEY>" "http://127.0.0.1:8000/filter?district=central"

# Get dataset info
curl http://127.0.0.1:8000/info
This implementation ensures consistent, secure, and efficient access to district-level datasets, while providing a foundation for future enhancements such as integrating additional data sources, expanding temporal coverage, and improving API functionality.

##	Future Enhancement
For future improvements, several enhancements are planned:
1.	Expanded Master Lookup: Incorporate newly formed districts and administrative regions, and continuously update the reference table to maintain data accuracy.
2.	Automated Name Standardization: Use fuzzy matching or machine learning-based entity resolution to reduce manual effort in reconciling inconsistent names across datasets.
3.	Extended Temporal Coverage: Include data beyond 2017 and integrate real-time data sources through APIs to enable trend analysis and timely reporting.
4.	Enhanced API Functionality: Add endpoints for aggregated statistics, advanced filtering, and data visualization to improve usability for stakeholders.
5.	Scalability and Deployment: Deploy the API on cloud platforms with load balancing and caching to handle larger datasets and higher concurrent traffic.
These steps aim to make the District Data API a robust, reliable, and extensible platform for researchers, policymakers, and developers working with district-level datasets in Pakistan.


2. NLP Model Training for Urdu
The NLP component required text classification capabilities in Urdu. Two datasets were used for model training:
- Publicly available Urdu dataset (limited to sentiment analysis)
- A custom AI-generated dataset with multiple labels for broader classification tasks

The limitation of the public dataset was its narrow focus, which led to the creation of a new dataset (released publicly under an open license for community use): https://huggingface.co/datasets/umar178/UrduMultiDomainClassification

Models were fine-tuned and published on Hugging Face for open access: https://huggingface.co/umar178/UrduTextClassificationModels/tree/main


3. API Development
To enable practical use of the trained models, a RESTful API layer was developed and deployed on a private VPS. The APIs were secured with authentication. Three main endpoints were created:
- getanalysis: Sentiment analysis
- getintent: Intent classification
- gettopic: Topic classification

All endpoints accept a JSON request body containing the input message and return the predicted label along with a confidence score.
On top of that the project is also hosted on a private vps allowing for public api calls.

# Sample API Call
curl -X POST "http://n8n.srv940619.hstgr.cloud:5050/gettopic" ^
  -H "Content-Type: application/json" ^
  -H "Authorization: Bearer <token>" ^
  -d "{\"message\":\"آپ کیسے ہیں\"}"
where bearer token is: 9f4b2c6d87a3e015d2e9c84b7f61a1b8c3d74e5f00a19d8f6b42e7d1ac59f304

# Results
The project successfully met all technical requirements of Phase 1:
- Designed reusable schemas for government datasets.
- Trained and fine-tuned Urdu NLP models with accuracy exceeding 80%.
- Developed and deployed secure RESTful APIs with three functional endpoints.

This infrastructure demonstrates the feasibility of localized AI infrastructure for Pakistan.

# Relevance for Pakistan
This project directly supports Pakistan’s digital independence by:
- Enabling AI applications in Urdu and regional languages.
- Providing reusable data schemas to standardize government datasets.
- Offering open-source NLP models and datasets for public and private sector innovation.
- Supporting the Pakistan Digital Authority Act and National AI Policy.

# Conclusion
Phase 1 successfully solved the foundational problems of schema design, localized NLP model training, and secure API development. The outputs are reusable, open, and ready to serve as the basis for Phase 2 and eventual commercialization as the Pakistan National AI Cloud & API Gateway.
