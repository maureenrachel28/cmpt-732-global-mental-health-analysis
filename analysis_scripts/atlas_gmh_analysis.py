import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

from pyspark.sql import SparkSession, types
from pyspark.sql import functions as F

from utils.udf import convert_to_score, convert_to_binary

spark = SparkSession.builder.appName('BDL Project - Team ML').getOrCreate()
assert spark.version >= '3.0' # make sure we have Spark 3.0+
spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext

# Helper function
def save_pandas_csv(df, location, filename):
    df.to_csv(location + "/" + filename + '.csv', index=False)
    
def run():
    
    # Atlas countries facts
    atlas_countries_facts_schema = types.StructType([
        types.StructField("facts_country_name", types.StringType()),
        types.StructField("facts_country_code", types.StringType()),
        types.StructField("facts_un_region", types.StringType()),
    ])

    atlas_countries_facts = spark.read.csv("datasets_output/atlas_fact_dims/atlas_countries_facts/", schema=atlas_countries_facts_schema)
    atlas_countries_facts.cache()
    
    # Atlas basic info
    atlas_countries_basic_info_schema = types.StructType([
        types.StructField("basic_country_code", types.StringType()),
        types.StructField("basic_population", types.IntegerType()),
        types.StructField("basic_income_group", types.StringType()),
        types.StructField("basic_who_region", types.StringType()),
        types.StructField("basic_expenditure_cad", types.FloatType()),
    ])

    atlas_countries_basic_info = spark.read.csv("datasets_output/atlas_fact_dims/atlas_countries_basic_info_dims/", schema=atlas_countries_basic_info_schema)
    
    # Countries with basic info
    atlas_basic_combined = atlas_countries_basic_info.join(atlas_countries_facts.hint("broadcast"), (atlas_countries_basic_info["basic_country_code"]==atlas_countries_facts["facts_country_code"]))
    atlas_basic_combined.cache()
    
    # Atlas smr dataset
    atlas_smr_schema = types.StructType([
        types.StructField("smr_country_code", types.StringType()),
        types.StructField("smr_suicide_mortality_rate_2013", types.FloatType()),
        types.StructField("smr_suicide_mortality_rate_2016", types.FloatType()),
        types.StructField("smr_suicide_mortality_rate_2019", types.FloatType()),
    ])

    atlas_smr_df = spark.read.option("multiline", "true").csv("datasets_output/atlas_fact_dims/atlas_smr_info_dims", schema=atlas_smr_schema)
    
    # Combine smr with basic combined info
    smr_basic_combined = atlas_basic_combined.join(atlas_smr_df.hint("broadcast"), (atlas_basic_combined["basic_country_code"]==atlas_smr_df["smr_country_code"])).drop("basic_country_code")
    smr_basic_combined.cache()
    smr_basic_combined_temp = smr_basic_combined.select([
        "basic_expenditure_cad",
        "facts_country_name",
        "smr_suicide_mortality_rate_2019"
    ]).withColumnsRenamed({
        "basic_expenditure_cad": "expenditure_cad",
        "facts_country_name": "country_name",
        "smr_suicide_mortality_rate_2019": "suicide_mortality_rate"
    })
    
    smr_df_pd = smr_basic_combined_temp.toPandas()
    
    save_pandas_csv(smr_df_pd, "csv_out/atlas+gmh", "smr_df_pd")
    
    # Read suicide dataset from GMH
    sui_schema = types.StructType([
        types.StructField("sui_country_name", types.StringType()),
        types.StructField("sui_country_code", types.StringType()),
        types.StructField("sui_un_region", types.StringType()),
        types.StructField("sui_year", types.IntegerType()),
        types.StructField("sui_male_suicide_%", types.FloatType()),
        types.StructField("sui_female_suicide_%", types.FloatType()),
    ])
    sui_mf_df = spark.read.option("multiline", "true").csv("datasets_output/gmh_fact_dims/sui_mf", schema=sui_schema)
    
    # Read atlas dataset programs section
    atlas_programs_schema = types.StructType([
        types.StructField("program_suicide_prevention_program", types.StringType()),
        types.StructField("program_awareness_anti_stigma_program", types.StringType()),
        types.StructField("program_early_child_development_program", types.StringType()),
        types.StructField("program_school_based_program", types.StringType()),
        types.StructField("program_parental_health_program", types.StringType()),
        types.StructField("program_work_related_program", types.StringType()),
        types.StructField("program_disaster_preparation_program", types.StringType()),
        types.StructField("program_country_code", types.StringType())
    ])
    atlas_programs_df = spark.read.option("multiline", "true").csv("datasets_output/atlas_fact_dims/atlas_countries_programs_info_dims", schema=atlas_programs_schema)
    
    # Fill in no_program for countries with no program (null value)
    atlas_programs_df.fillna("no_program")
    
    # Combine programs with basic combined info
    atlas_programs_combined = atlas_programs_df.join(atlas_countries_facts.hint("broadcast"), (atlas_programs_df["program_country_code"]==atlas_countries_facts["facts_country_code"]))
    atlas_programs_combined.cache()
    
    # Combine with sui_mf_df
    atlas_suicide_programs_sui = atlas_programs_combined.select(["program_country_code", "program_suicide_prevention_program"]).join(sui_mf_df.hint("broadcast"), (atlas_programs_combined["program_country_code"]==sui_mf_df["sui_country_code"]))
    atlas_suicide_programs_sui = atlas_suicide_programs_sui.fillna("no_program")
    
    # Filter by existence or lack of suicide prevention program
    no_suicide_programs_sui = atlas_suicide_programs_sui.filter(atlas_suicide_programs_sui["program_suicide_prevention_program"]=="no_program")
    suicide_programs_sui = atlas_suicide_programs_sui.filter(~(atlas_suicide_programs_sui["program_suicide_prevention_program"]=="no_program"))
    
    no_suicide_programs_sui = no_suicide_programs_sui.select([
        "sui_country_name",
        "sui_male_suicide_%",
        "sui_female_suicide_%"
    ]).withColumnsRenamed({
        "sui_country_name": "country_name",
        "sui_male_suicide_%": "male_suicide_%",
        "sui_female_suicide_%": "female_suicide_%",
    })
    
    suicide_programs_sui = suicide_programs_sui.select([
        "sui_country_name",
        "sui_male_suicide_%",
        "sui_female_suicide_%"
    ]).withColumnsRenamed({
        "sui_country_name": "country_name",
        "sui_male_suicide_%": "male_suicide_%",
        "sui_female_suicide_%": "female_suicide_%",
    })
    
    # Suicide rates of top 5 suicide countries by male and female without program
    top_no_program_male = no_suicide_programs_sui.orderBy(F.desc("male_suicide_%")).limit(5)
    top_no_program_female = no_suicide_programs_sui.orderBy(F.desc("female_suicide_%")).limit(5)
    top_no_program = top_no_program_male.union(top_no_program_female).dropDuplicates()
    
    # Suicide rates of top 5 suicide countries by male and female without program
    top_program_male = suicide_programs_sui.orderBy(F.desc("male_suicide_%")).limit(5)
    top_program_female = suicide_programs_sui.orderBy(F.desc("female_suicide_%")).limit(5)
    top_program = top_program_male.union(top_program_female).dropDuplicates()
    
    # Union both
    combined_top_program_no_program = top_program.union(top_no_program)
    
    combined_avg_program_no_program = atlas_suicide_programs_sui.withColumn(
        "program_status",
        F.when((atlas_suicide_programs_sui["program_suicide_prevention_program"] == "no_program"),
            "No suicide prevention program").otherwise("Suicide prevention program")
    ).select(["program_status", "sui_male_suicide_%", "sui_female_suicide_%"]).groupBy("program_status").agg(
        F.avg(F.col("sui_male_suicide_%")),
        F.avg(F.col("sui_female_suicide_%"))
    )
    
    combined_avg_program_no_program_pd_df = combined_avg_program_no_program.toPandas()
    
    save_pandas_csv(combined_avg_program_no_program_pd_df, "csv_out/atlas+gmh", "combined_avg_program_no_program_pd_df")
    
    # Atlas MH expenditure
    atlas_exp_and_pay_facts_schema = types.StructType([
        types.StructField("ep_country_code", types.StringType()),
        types.StructField("govt_exp_mental_health_%_budget", types.FloatType()),
        types.StructField("mh_expenditure_hospital", types.FloatType()),
        types.StructField("pay_for_services", types.StringType()),
        types.StructField("pay_for_medication", types.StringType()),
        types.StructField("insurance_and_reimbursement_includes_mental_health", types.StringType()),
    ])

    atlas_exp_and_pay_df = spark.read.csv("datasets_output/atlas_fact_dims/atlas_exp_and_pay_info_dims/", schema=atlas_exp_and_pay_facts_schema)
    
    atlas_exp_and_pay_df = atlas_exp_and_pay_df.fillna(-1, subset=["govt_exp_mental_health_%_budget", "mh_expenditure_hospital"]).fillna("no_info", subset=["pay_for_services", "pay_for_medication"]).withColumn("insurance_and_reimbursement_includes_mental_health", F.lower("insurance_and_reimbursement_includes_mental_health")).fillna("no_info", subset=["insurance_and_reimbursement_includes_mental_health"])
    
    # Atlas mental health workers dataset
    atlas_mental_health_workers_schema = types.StructType([
        types.StructField("mh_country_code", types.StringType()),
        types.StructField("mh_num_psychiatrists", types.FloatType()),
        types.StructField("mh_num_nurses", types.FloatType()),
        types.StructField("mh_num_psychologists", types.FloatType()),
        types.StructField("mh_num_social_workers", types.FloatType()),
        types.StructField("mh_num_other_specialized_workers", types.FloatType()),
        types.StructField("mh_num_total_mental_health_workers", types.FloatType()),
        types.StructField("mh_num_total_mental_health_workers_2014", types.FloatType()),
        types.StructField("mh_num_total_mental_health_workers_2017", types.FloatType()),
        types.StructField("mh_num_total_mental_health_workers_2020", types.FloatType()),
        types.StructField("mh_num_child_psychiatrists", types.FloatType()),
        types.StructField("mh_num_child_mental_health_workers", types.FloatType()),
    ])

    atlas_mental_health_workers_df = spark.read.csv("datasets_output/atlas_fact_dims/atlas_mental_health_workers_info_dims/", schema=atlas_mental_health_workers_schema)
    
    # Fill unreported values (null) with -1
    atlas_mental_health_workers_df = atlas_mental_health_workers_df.fillna(-1, subset=["mh_num_psychiatrists", "mh_num_nurses", "mh_num_psychologists", "mh_num_social_workers", "mh_num_other_specialized_workers", "mh_num_total_mental_health_workers", "mh_num_total_mental_health_workers_2014", "mh_num_total_mental_health_workers_2017", "mh_num_total_mental_health_workers_2020", "mh_num_child_psychiatrists", "mh_num_child_mental_health_workers"])
    
    # Compare govt expenditure to number of mental health workers
    exp_vs_social_workers_df = atlas_exp_and_pay_df.select(["ep_country_code", "govt_exp_mental_health_%_budget"]).join(atlas_mental_health_workers_df.select(["mh_country_code", "mh_num_total_mental_health_workers"]).hint("broadcast"), (atlas_exp_and_pay_df["ep_country_code"]==atlas_mental_health_workers_df["mh_country_code"])).drop("mh_country_code")
    exp_vs_social_workers_df = exp_vs_social_workers_df.join(atlas_basic_combined.select(["basic_country_code", "basic_population", "basic_income_group", "facts_country_name"]).hint("broadcast"), (atlas_basic_combined["basic_country_code"]==exp_vs_social_workers_df["ep_country_code"]))
    exp_vs_social_workers_df = exp_vs_social_workers_df.drop("ep_country_code")
    
    exp_vs_social_workers_pd_df = exp_vs_social_workers_df.toPandas()
    
    save_pandas_csv(exp_vs_social_workers_pd_df, "csv_out/atlas+gmh", "exp_vs_social_workers_pd_df")
    
    # Atlas facilities dataset
    atlas_facilities_schema = types.StructType([
        types.StructField("facilities_country_code", types.StringType()),
        types.StructField("outpatient_facilities_attached_to_hospitals", types.FloatType()),
        types.StructField("outpatient_facilities_not_attached_to_hospitals", types.FloatType()),
        types.StructField("other_outpatient_facilities", types.FloatType()),
        types.StructField("children_specific_outpatient_facilities", types.FloatType()),
        types.StructField("inpatient_hospitals", types.FloatType()),
        types.StructField("inpatient_psychiatric_units", types.FloatType()),
        types.StructField("community_residential_facilities", types.FloatType()),
        types.StructField("children_specific_inpatient_facilities", types.FloatType()),
        types.StructField("mental_hospital_beds", types.FloatType()),
        types.StructField("psych_bed", types.FloatType()),
        types.StructField("community_beds", types.FloatType()),
        types.StructField("children_specific_beds", types.FloatType()),
        types.StructField("total_community_facilities", types.FloatType()),
    ])

    atlas_facilities_info = spark.read.csv("datasets_output/atlas_fact_dims/atlas_mental_health_facilities_info_dims/", schema=atlas_facilities_schema)
    
    # Fill unreported values (null) with -1
    atlas_facilities_info = atlas_facilities_info.fillna(-1)
    
    # To assign scores to the countries, we want to take i*10-th percentiles and then sum for all facilities
    # Check UDF for score assigned
    column_percentiles = {}
    for column in atlas_facilities_info.columns:
        if column == "facilities_country_code":
            continue
        column_percentiles[column] = atlas_facilities_info.approxQuantile(column, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 0.001)
    
    # Score important parameters based on percentile    
    atlas_facilities_info_scored = atlas_facilities_info
    for column in atlas_facilities_info.columns:
        if column in ["facilities_country_code", "mental_hospital_beds", "psych_bed", "community_beds", "children_specific_beds"]:
            continue
        atlas_facilities_info_scored = atlas_facilities_info_scored.withColumn(column+"_score", convert_to_score(column, F.array(F.lit(column_percentiles[column]))))
    
    # Sum of individual scores
    atlas_facilities_info_scored = atlas_facilities_info_scored.withColumn("total_facilities_score", F.col("outpatient_facilities_attached_to_hospitals_score")+F.col("children_specific_outpatient_facilities_score")+F.col("inpatient_hospitals_score")+F.col("inpatient_psychiatric_units_score")+F.col("community_residential_facilities_score")+F.col("children_specific_inpatient_facilities_score")+F.col("total_community_facilities_score")).select(["facilities_country_code", "total_facilities_score"])
    
    # Write 1 if there exists a program and 0 otherwise, this will help us to compute the score
    atlas_programs_combined_binarized = atlas_programs_combined
    for column in atlas_programs_combined_binarized.columns:
        if column in ["program_country_code", "facts_country_name", "facts_country_code", "facts_un_region"]:
            continue
        atlas_programs_combined_binarized = atlas_programs_combined_binarized.withColumn(column, convert_to_binary(F.col(column)))
        
    # Take sum of program scores
    atlas_programs_scored = atlas_programs_combined_binarized.withColumn("total_program_score", F.col("program_suicide_prevention_program")+F.col("program_awareness_anti_stigma_program")+F.col("program_early_child_development_program")+F.col("program_school_based_program")+F.col("program_parental_health_program")+F.col("program_work_related_program")+F.col("program_disaster_preparation_program")).select(["program_country_code", "facts_country_name", "total_program_score", "facts_un_region"])
    
    # Try to convert govt expenditure to score as well
    atlas_mh_expenditure_temp = atlas_exp_and_pay_df.select(["ep_country_code", "govt_exp_mental_health_%_budget"])
    # Use the same percentile approach
    column_percentiles["govt_exp_mental_health_%_budget"] = atlas_mh_expenditure_temp.approxQuantile("govt_exp_mental_health_%_budget", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 0.001)
    atlas_mh_expenditure_temp = atlas_mh_expenditure_temp.withColumn("total_govt_exp_mental_health_%_budget_score", convert_to_score("govt_exp_mental_health_%_budget", F.array(F.lit(column_percentiles["govt_exp_mental_health_%_budget"])))).select(["ep_country_code", "total_govt_exp_mental_health_%_budget_score"])
    
    # Calculate and store final score for each country
    final_score_df = atlas_facilities_info_scored.join(atlas_programs_scored.hint("broadcast"), (atlas_facilities_info_scored["facilities_country_code"]==atlas_programs_scored["program_country_code"])).join(atlas_mh_expenditure_temp, (atlas_facilities_info_scored["facilities_country_code"]==atlas_mh_expenditure_temp["ep_country_code"]))
    final_score_df = final_score_df.withColumn("total_score", F.col("total_facilities_score")+F.col("total_program_score")+F.col("total_govt_exp_mental_health_%_budget_score")).select(["facts_country_name", "facts_un_region", "total_score"])
    final_score_df_pd = final_score_df.toPandas()
    save_pandas_csv(final_score_df_pd, "csv_out/atlas+gmh", "final_score_df_pd")
    
    # Find top and bottom 5 countries by score
    final_score_df = final_score_df.orderBy(F.desc(F.col("total_score")))
    top_5_countries = final_score_df.limit(5)
    bottom_5_countries = spark.createDataFrame(final_score_df.tail(5))
    
    top_bottom_countries = top_5_countries.union(bottom_5_countries)
    
    top_bottom_countries_pd_df = top_bottom_countries.toPandas()
    
    save_pandas_csv(top_bottom_countries_pd_df, "csv_out/atlas+gmh", "top_bottom_countries_pd_df")