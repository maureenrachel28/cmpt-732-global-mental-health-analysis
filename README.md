# cmpt-732-project-ml

# Storytelling with Data: Mental Health Through The Data Lens

### Live site:  http://35.91.214.134:5000/

Frequently, the hesitation to seek mental health support is anchored in the societal stigma surrounding such conditions. A staggering 70% of individuals have expressed their reluctance to seek help or confide due to this stigma. Ensuring a safe space for people suffering from mental health problems requires awareness of the problems themselves as well as the stigma surrounding them. However, it is often difficult to find data on stigma or how it propagates; most datasets are around statistics on mental health indicators or problems. As such, most mental health programs end up being non-data-driven efforts or do not factor in the effects of stigma. 

The solution is to build effective anti-stigma and awareness campaigns and programs with the analysis presented and address the crucial challenge of the lack of a consistent, standardized mental health metrics reporting framework


Our goal with this project is to offer valuable insights to stakeholders (NGOs, mental health professionals, campaign coordinators, and mental health program creators) and enable them to create these campaigns. Our analysis is built around the prevalence of mental health disorders, infrastructure available to address them, and discussions around mental health on social media to uncover stigma and keywords.

We use PySpark and Spark MLLib for our analysis, Pandas and Plotly for our visualizations and finally host the Flask app on AWS EC2 (linked above).

We use the following datasets:
<ol>
<li><a href=https://www.kaggle.com/datasets/kamaumunyori/global-health-data-analysis-1990-2019>Global Health Data Analysis 1990-2019</a>
</li>
<li><a href=https://www.who.int/teams/mental-health-and-substance-use/data-research/mental-health-atlas>WHO Mental Health Project Atlas</a>
</li>
<li><a href=https://zenodo.org/records/3941387>Reddit Mental Health Dataset</a></li>
</ol>

Project structure:
<ul>
<li>analysis_scripts: Includes scripts to run analysis on Atlas and Global Health Data Analysis datasets.
</li>
<li>csv_out: Dummy folder that needs to be uploaded to HDFS when running on the cluster and also includes Pandas CSVs that we use for visualization (these can be ignored).
</li>
<li>datasets_input: Also needs to be uploaded to the HDFS when running on the cluster. Includes Atlas dataset and Global Health Data Analysis dataset CSVs.
</li>
<li>etl_scripts: Includes ETL scripts for Atlas and Global Health Data Analysis datasets.
</li>
<li>experimental_notebooks: Includes experimental Jupyter notebooks on ETL, Analysis and Visualization for Atlas and Global Health Data Analysis datasets.
</li>
<li>MH Flask: Includes code for the frontend Flask app.</li>
<li>scrape: Includes experimental code for extracting data from the Atlas dataset.</li>
<li>topic-modelling: Includes Colaboratory notebooks for the Reddit Mental Health Data analysis and visualization.</li>
<li>utils: Includes UDFs for Atlas and Global Health Data Analysis datasets.</li>
<li>main.py: Main app code for Atlas and Global Health Data Analysis portion.</li>
</ul>

NOTE: Ahmad Al Fayad Chowdhury (aac22) shows up in the commits, but does not show up as a contributor. Please take into account his commits.