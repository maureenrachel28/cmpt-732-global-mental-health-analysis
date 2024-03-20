import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

from pyspark.sql import SparkSession, types
from pyspark.sql import functions as F

from utils.udf import convert_population, convert_who_region, convert_income_group, extract_exp, extract_code, convert_smr, convert_dash_to_null_string, convert_authority, convert_dash_to_null_float, convert_payment, convert_percentage_to_category_float, convert_dash_to_null_table_string

spark = SparkSession.builder.appName('BDL Project - Team ML').getOrCreate()
assert spark.version >= '3.0' # make sure we have Spark 3.0+
spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext


# Helper functions
def read_input(input_dir):
    return spark.read.option("header", True).option("multiline", "true").csv(input_dir) # header parsed, multiline = True helps with multiline values

def save_output(df, output_dir):
    df.write.csv(output_dir, mode="overwrite")
    

# Main function
def run(input_dir_atlas, output_dir_atlas):
    
    # Define input/output directories
    input_data_dir = input_dir_atlas + "/atlas_csv/"
    output_data_dir = output_dir_atlas + "/atlas_fact_dims/"

    # Atlas Countries
    input_mhdb = input_data_dir + "MH_Database_Combined_Countries.csv"
    # Atlas Programs
    input_mhdb_tables = input_data_dir + "MH_Database_Table_Combined.csv"
    # Country dict
    input_atlas_countries_dict = input_dir_atlas + "/atlas_country_dict"
    # Currency rates
    input_currency_rates = input_data_dir + "rates.csv"
    # Output fact-dimension tables
    output_atlas_countries_facts = output_data_dir + "atlas_countries_facts"
    output_atlas_countries_basic_info_dims = output_data_dir + "atlas_countries_basic_info_dims"
    output_atlas_smr_info_dims = output_data_dir + "atlas_smr_info_dims"
    # Don't need this
    # output_atlas_policies_info_dims = output_data_dir + "atlas_policies_info_dims"
    output_atlas_exp_and_pay_info_dims = output_data_dir + "atlas_exp_and_pay_info_dims"
    output_atlas_mental_health_workers_info_dims = output_data_dir + "atlas_mental_health_workers_info_dims"
    # Don't need this
    # output_atlas_patient_admission_info_dims = output_data_dir + "atlas_patient_admission_info_dims"
    output_atlas_mental_health_facilities_info_dims = output_data_dir + "atlas_mental_health_facilities_info_dims"
    output_atlas_countries_programs_info_dims = output_data_dir + "atlas_countries_programs_info_dims"
    
    # Read entire countries dataset
    countries = read_input(input_mhdb)
    
    # Read country dictionary
    atlas_countries_dict_schema = types.StructType([
        types.StructField("country_dict_country", types.StringType()),
        types.StructField("country_dict_country_name", types.StringType()),
        types.StructField("country_dict_country_code", types.StringType()),
        types.StructField("country_dict_un_region", types.StringType()),
    ])
    atlas_countries_dict = spark.read.csv(input_atlas_countries_dict, schema = atlas_countries_dict_schema)
    
    # Merge with atlas countries dict to get country_code, country_name (official name), un_region
    countries = countries.join(atlas_countries_dict.hint("broadcast"), (countries["country"]==atlas_countries_dict["country_dict_country"]), how="left")
    
    countries = countries.drop("country").withColumnsRenamed({
        "country_dict_country": "country",
        "country_dict_country_name": "country_name",
        "country_dict_country_code": "country_code",
        "country_dict_un_region": "un_region"
    })
    countries = countries.repartition(41) # If there are 164 countries, try to deal with 4 countries per partition
    countries.cache() # Cache here because this gets used a lot
    
    # # Augment country_code, country_name (official name), un_region
    # countries = countries.withColumn("country_code", get_country_code(F.col("country"))).withColumn("country_name", get_country_name(F.col("country"))).withColumn("un_region", get_country_region(F.col("country")))
    
    # Form and save country fact table for country_name, country_code, un_region
    atlas_countries_df = countries.select([
        "country_name",
        "country_code",
        "un_region"
    ])
    save_output(atlas_countries_df, output_atlas_countries_facts)
    
    # Form and save country basic info dim table
    atlas_countries_basic_info_df = countries.select([
        "country_code",
        "population",
        "income_group",
        "expenditure($)",
        "who_region"
    ])
    atlas_countries_basic_info_df.cache() # Cache this because there are several operations on it
    
    # Convert population to float, who_region and income_group to lower_case strings
    atlas_countries_basic_info_df = atlas_countries_basic_info_df.withColumn("population", convert_population(F.col("population")))
    atlas_countries_basic_info_df = atlas_countries_basic_info_df.withColumn("who_region", convert_who_region(F.col("who_region")))
    atlas_countries_basic_info_df = atlas_countries_basic_info_df.withColumn("income_group", convert_income_group(F.col("income_group")))
    
    # Convert expenditure amounts to CAD by splitting amount and currency and then converting to CAD
    atlas_countries_basic_info_df = atlas_countries_basic_info_df.withColumn("expenditure_amount", extract_exp(F.col("expenditure($)")))
    atlas_countries_basic_info_df = atlas_countries_basic_info_df.withColumn("expenditure_currency", extract_code(F.col("expenditure($)")))
    
    rates_schema = types.StructType([
        types.StructField('currency', types.StringType()),
        types.StructField('rate', types.StringType()),
    ])

    rates_df = spark.read.option("header", True).csv(input_currency_rates, schema=rates_schema)
    
    # To merge with currency conversion rates
    atlas_basic_info_df_currency = atlas_countries_basic_info_df.join(rates_df.hint("broadcast"), (atlas_countries_basic_info_df["expenditure_currency"]==rates_df["currency"]), how="left")
    atlas_basic_info_df_currency.cache() # Cache this because there are a few operations on it
    
    # Collect CAD rate
    rate = atlas_basic_info_df_currency.filter(atlas_basic_info_df_currency["expenditure_currency"]=="CAD").collect()[0]["rate"]
    
    # Convert to CAD
    atlas_basic_info_df_currency = atlas_basic_info_df_currency.withColumn("expenditure_cad", F.col("expenditure_amount")/F.col("rate")*rate)
    
    atlas_basic_info_final_df = atlas_basic_info_df_currency.select([
        "country_code",
        "population",
        "income_group",
        "who_region",
        "expenditure_cad"
    ])
    
    save_output(atlas_basic_info_final_df, output_atlas_countries_basic_info_dims)
    
    # Form and save suicide mortality rate info dim table
    smr_info_df = countries.select([
        "country_code",
        "smr_2013",
        "smr_2016",
        "smr_2019"
    ])
    
    smr_info_df = smr_info_df.withColumn("suicide_mortality_rate_2013", convert_smr(F.col("smr_2013"))).withColumn("suicide_mortality_rate_2016", convert_smr(F.col("smr_2016"))).withColumn("suicide_mortality_rate_2019", convert_smr(F.col("smr_2019"))).drop("smr_2013").drop("smr_2016").drop("smr_2019")
    
    smr_info_df.write.csv(output_atlas_smr_info_dims, mode='overwrite')
    
    # Don't need this
    # # Form and save policy info dim table
    # policies_info_df = countries.select([
    #     "country_code",
    #     "policy",
    #     "year_policy",
    #     "hr_resources_plan",
    #     "finance_resources_plan",
    #     "mh_law",
    #     "law_year",
    #     "dedcated_body",
    #     "mh_child_policy",
    #     "year_child_policy",
    #     "mh_adolescent_policy",
    #     "year_adolescent_policy",
    #     "mh_suicide_prevention_policy",
    #     "year_suicide_prevention_policy"
    # ])
    # policies_info_df.cache() # Cache this because there are several operations on it
    
    # policies_info_df = policies_info_df.withColumnsRenamed({
    #     "policy": "mental_health_policy",
    #     "year_policy": "mental_health_policy_year",
    #     "hr_resources_plan": "human_resources_for_mental_health_policy",
    #     "finance_resources_plan": "finance_resources_for_mental_health_policy",
    #     "mh_law": "mental_health_law",
    #     "law_year": "mental_health_law_year",
    #     "dedcated_body": "dedicated_authority",
    #     "mh_child_policy": "mental_health_child_policy",
    #     "year_child_policy": "mental_health_child_policy_year",
    #     "mh_adolescent_policy": "mental_health_adolescent_policy",
    #     "year_adolescent_policy": "mental_health_adolescent_policy_year",
    #     "mh_suicide_prevention_policy": "suicide_prevention_policy",
    #     "year_suicide_prevention_policy": "suicide_prevention_policy_year"
    # })
    
    # for col in policies_info_df.columns:
    #     policies_info_df = policies_info_df.withColumn(col, convert_dash_to_null_string(F.col(col)))
        
    # replacement_pattern = "[\\n\\r]"
    # policies_info_df = policies_info_df.withColumn("dedicated_authority", F.regexp_replace("dedicated_authority", replacement_pattern, " "))
    # policies_info_df = policies_info_df.withColumn("dedicated_authority", convert_authority(F.col("dedicated_authority")))
    # save_output(policies_info_df, output_atlas_policies_info_dims)
    
    # Form and save government expenditure info dim table
    exp_and_pay_info_df = countries.select([
        "country_code",
        "mh_expenditure",
        "mh_expenditure_hospital",
        "service_pay_method",
        "service_pay_medication",
        "mh_inclusive_schemes"
    ])
    exp_and_pay_info_df.cache() # Cache this because there are several operations on it
    
    exp_and_pay_info_df = exp_and_pay_info_df.withColumn("mh_expenditure", convert_dash_to_null_float(F.col("mh_expenditure"))).withColumn("mh_expenditure_hospital", convert_dash_to_null_float(F.col("mh_expenditure_hospital"))).withColumn("mh_inclusive_schemes", convert_dash_to_null_string(F.col("mh_inclusive_schemes")))
    
    replacement_pattern = "[\\n\\r]"
    
    exp_and_pay_info_df = exp_and_pay_info_df.withColumn("service_pay_method", F.regexp_replace("service_pay_method", replacement_pattern, "")).withColumn("service_pay_medication", F.regexp_replace("service_pay_medication", replacement_pattern, ""))
    
    exp_and_pay_info_df = exp_and_pay_info_df.withColumn("service_pay_medication", convert_payment(F.col("service_pay_medication"))).withColumn("service_pay_method", convert_payment(F.col("service_pay_method")))
    
    exp_and_pay_info_df = exp_and_pay_info_df.withColumnsRenamed({
        "mh_expenditure": "govt_exp_mental_health_%_budget",
        "mh_hospital": "govt_exp_mental_health_%_hospital",
        "service_pay_method": "pay_for_services",
        "service_pay_medication": "pay_for_medication",
        "mh_inclusive_schemes": "insurance_and_reimbursement_includes_mental_health"
    })
    
    save_output(exp_and_pay_info_df, output_atlas_exp_and_pay_info_dims)
    
    # Form and save mental health workers info dim table
    mental_health_workers_info_df = countries.select([
        "country_code",
        "no_psychiatrists",
        "no_total_health_profs",
        "no_psychologists",
        "no_social_wrks",
        "no_other_workers",
        "no_total_workers",
        "mh_workers_2014",
        "mh_workers_2017",
        "mh_workers_2020",
        "mh_workers_psych",
        "mh_workers_all"
    ])
    mental_health_workers_info_df.cache() # Cache this because there are several operations on it
    
    for col in mental_health_workers_info_df.columns:
        if col == "country_code":
            continue
        mental_health_workers_info_df = mental_health_workers_info_df.withColumn(col, convert_dash_to_null_float(F.col(col)))
        
    mental_health_workers_info_df = mental_health_workers_info_df.withColumnsRenamed({
        "no_psychiatrists": "num_psychiatrists",
        "no_total_health_profs": "num_nurses",
        "no_psychologists": "num_psychologists",
        "no_social_wrks": "num_social_workers",
        "no_other_workers": "num_other_specialized_workers",
        "no_total_workers": "num_total_mental_health_workers",
        "mh_workers_2014": "num_total_mental_health_workers_2014",
        "mh_workers_2017": "num_total_mental_health_workers_2017",
        "mh_workers_2020": "num_total_mental_health_workers_2020",
        "mh_workers_psych": "num_child_psychiatrists",
        "mh_workers_all": "num_child_mental_health_workers"
    })
    
    save_output(mental_health_workers_info_df, output_atlas_mental_health_workers_info_dims)
    
    # Don't need this
    # # Form and save patient admission info dim table
    # patient_admissions_info_df = countries.select([
    #     "country_code",
    #     "ip_adm_2014",
    #     "ip_adm_2017",
    #     "ip_adm_2020",
    #     "op_adm_2014",
    #     "op_adm_2017",
    #     "op_adm_2020",
    #     "ca_services_2014",
    #     "ca_services_2017",
    #     "ca_services_2020",
    #     "hosp_facilities_vists",
    #     "non_hosp_facilities_visits",
    #     "other_facilities_vists",
    #     "op_facilities_ca_visits",
    #     "mh_annual",
    #     "psych_admission",
    #     "community_admissions",
    #     "ca_adminssions",
    #     "mh_hosps_total_adm",
    #     "involuntary_adms",
    #     "follow_up",
    #     "ip_lt_1yr",
    #     "ip_1_5",
    #     "ip_gt_5",
    #     "ip_timely_servies",
    #     "total_pshycosis",
    #     "total_pshycosis_male",
    #     "total_pshycosis_female"
    # ])
    # patient_admissions_info_df.cache() # Cache this because there are several operations on it
    
    # for col in patient_admissions_info_df.columns:
    #     if col == "country_code" or col == "follow_up" or col == "ip_timely_servies":
    #         continue
    #     patient_admissions_info_df = patient_admissions_info_df.withColumn(col, convert_dash_to_null_float(F.col(col)))
    
    # patient_admissions_info_df = patient_admissions_info_df.withColumn("follow_up", convert_percentage_to_category_float(F.col("follow_up"))).withColumn("ip_timely_servies", convert_percentage_to_category_float(F.col("ip_timely_servies")))
    
    # patient_admissions_info_df = patient_admissions_info_df.withColumnsRenamed({
    #     "ip_adm_2014": "inpatient_admissions_2014",
    #     "ip_adm_2017": "inpatient_admissions_2017",
    #     "ip_adm_2020": "inpatient_admissions_2020",
    #     "op_adm_2014": "outpatient_admissions_2014",
    #     "op_adm_2017": "outpatient_admissions_2017",
    #     "op_adm_2020": "outpatient_admissions_2020",
    #     "ca_services_2014": "community_based_mental_health_services_2014",
    #     "ca_services_2017": "community_based_mental_health_services_2017",
    #     "ca_services_2020": "community_based_mental_health_services_2020",
    #     "hosp_facilities_vists": "hospital_attached_facilities_visits",
    #     "non_hosp_facilities_visits": "non_hospital_attached_facilities_visits",
    #     "op_facilities_ca_visits": "children_specific_facilities_visits",
    #     "other_facilities_vists": "other_facilities_visits",
    #     "mh_annual": "annual_mental_hospital_admissions",
    #     "involuntary_adms": "total_involuntary_admissions",
    #     "follow_up": "follow_up_post_discharge",
    #     "ip_lt_1yr": "inpatients_staying_less_1_year",
    #     "ip_1_5": "inpatients_staying_1_5_year",
    #     "ip_gt_5": "inpatients_staying_more_5_year",
    #     "ip_timely_servies": "received_timely_services",
    #     "total_pshycosis": "total_psychosis",
    #     "total_pshycosis_male": "total_psychosis_male",
    #     "total_pshycosis_female": "total_psychosis_female",
    # })
    
    # save_output(patient_admissions_info_df, output_atlas_patient_admission_info_dims)
    
    # Form and save mental health facilities info dim table
    atlas_mental_health_facilities_info_df = countries.select([
        "country_code",
        "mh_op_facilities_hosp",
        "non_hosp_op_facilities",
        "other_op_facilities",
        "op_facilities_ca",
        "ip_hosp",
        "ip_psych",
        "ip_community",
        "ip_ca",
        "mh_beds",
        "psych_bed",
        "community_beds",
        "caa_beds",
        "total_comm_facilities"
    ])
    
    atlas_mental_health_facilities_info_df = atlas_mental_health_facilities_info_df.withColumnsRenamed({
        "mh_op_facilities_hosp": "outpatient_facilities_attached_to_hospitals",
        "non_hosp_op_facilities": "outpatient_facilities_not_attached_to_hospitals",
        "other_op_facilities": "other_outpatient_facilities",
        "op_facilities_ca": "children_specific_outpatient_facilities",
        "ip_hosp": "inpatient_hospitals",
        "ip_psych": "inpatient_psychiatric_units",
        "ip_community": "community_residential_facilities",
        "ip_ca": "children_specific_inpatient_facilities",
        "mh_beds": "mental_hospital_beds",
        "psych_beds": "psychiatric_beds",
        "community_beds": "community_beds",
        "caa_beds": "children_specific_beds",
        "total_comm_facilities": "total_community_facilities"
    })
    
    save_output(atlas_mental_health_facilities_info_df, output_atlas_mental_health_facilities_info_dims)
    
    # Form and save atlas program info dim table
    atlas_countries_programs_info_df = spark.read.option("header", True).option("multiline", "true").csv(input_mhdb_tables)
    
    atlas_countries_programs_info_df = atlas_countries_programs_info_df.join(atlas_countries_dict.hint("broadcast"), (atlas_countries_programs_info_df["Country"]==atlas_countries_dict["country_dict_country"]), how="left")
    
    atlas_countries_programs_info_df = atlas_countries_programs_info_df.repartition(41) # There are 164 countries, do 4 per partition
    atlas_countries_programs_info_df.cache() # Cache this because there are several operations on it
    
    atlas_countries_programs_info_df = atlas_countries_programs_info_df.drop("Country").drop("country_dict_country").drop("country_dict_country_name").drop("country_dict_un_region").withColumnsRenamed({
        "Suicide prevention programme": "suicide_prevention_program",
        "Mental Health Awareness /Anti- stigma": "awareness_anti_stigma_program",
        "Early Child Development": "early_child_development_program",
        "School based mental health prevention and promotion": "school_based_program",
        "Parental / Maternal mental health promotion and prevention": "parental_health_program",
        "Work-related mental health prevention and promotion": "work_related_program",
        "Mental health and psychosocial component of disaster preparedness, disaster risk reduction": "disaster_preparation_program",
        "country_dict_country_code": "country_code",
    })
    
    atlas_countries_programs_info_df = atlas_countries_programs_info_df.drop("Country").withColumnsRenamed({
        "Suicide prevention programme": "suicide_prevention_program",
        "Mental Health Awareness /Anti- stigma": "awareness_anti_stigma_program",
        "Early Child Development": "early_child_development_program",
        "School based mental health prevention and promotion": "school_based_program",
        "Parental / Maternal mental health promotion and prevention": "parental_health_program",
        "Work-related mental health prevention and promotion": "work_related_program",
        "Mental health and psychosocial component of disaster preparedness, disaster risk reduction": "disaster_preparation_program"
    })
    
    for col in atlas_countries_programs_info_df.columns:
        if col == "country_code":
            continue
        atlas_countries_programs_info_df = atlas_countries_programs_info_df.withColumn(col, convert_dash_to_null_table_string(F.col(col)))
        
    save_output(atlas_countries_programs_info_df, output_atlas_countries_programs_info_dims)