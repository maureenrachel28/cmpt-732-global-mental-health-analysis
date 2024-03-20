import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

from pyspark.sql import SparkSession, types
from pyspark.sql import functions as F

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
def run(input_dir_gmh, output_dir_gmh):
    
    # Define input/output directories
    input_data_dir = input_dir_gmh + "/gmh_csv/"
    output_data_dir = output_dir_gmh + "/gmh_fact_dims/"
    
    # Country dict
    input_sui_countries_dict = input_dir_gmh + "/gmh_sui_2023_country_dict"

    # Global health data analysis inputs
    input_mhdd = input_data_dir + "/Mental_health_Depression_disorder_Data.csv"
    input_dep_mf = input_data_dir + "/prevalence-of-depression-males-vs-females.csv"
    input_ed_mf = input_data_dir + "/prevalence-of-eating-disorders-in-males-vs-females.csv"
    input_ad_mf = input_data_dir + "/prevalence-of-anxiety-disorders-males-vs-females.csv"
    input_sch_mf = input_data_dir + "/prevalence-of-schizophrenia-in-males-vs-females.csv"
    input_bpd_mf = input_data_dir + "/prevalence-of-bipolar-disorder-in-males-vs-females.csv"
    input_sui_mf = input_data_dir + "/suicide-rate-by-country-2023.csv"
    
    # Output fact-dimension tables
    output_mhdd = output_data_dir + "mhdd"
    output_dep_mf = output_data_dir + "dep_mf"
    output_ed_mf = output_data_dir + "ed_mf"
    output_ad_mf = output_data_dir + "ad_mf"
    output_sch_mf = output_data_dir + "sch_mf"
    output_bpd_mf = output_data_dir + "bpd_mf"
    output_sui_mf = output_data_dir + "sui_mf"
    
    # Read atlas countries fact table
    input_atlas_countries_facts = output_dir_gmh + "/atlas_fact_dims/atlas_countries_facts/"
    country_fact_schema = types.StructType([
        types.StructField('country_name', types.StringType()),
        types.StructField('country_code', types.StringType()),
        types.StructField('un_region', types.StringType()),
    ])
    atlas_countries_df = spark.read.csv(input_atlas_countries_facts, schema=country_fact_schema)
    
    atlas_countries_df.cache() # Cache this because there are several joins involving this
    
    # Mental health depression disorder data
    mhdd_df = spark.read.option("header", True).csv(input_mhdd)
    
    mhdd_df = mhdd_df.repartition(56) # There are 4424 rows, work on 79 rows per partition
    mhdd_df.cache() # Cache this because this is used a lot
    
    # Transformation 1: There are a lot of weird dates on the data. Some ranging all the way back to BCE. We want data only from 1990-2023 because it is more likely to be relevat and properly documented.
    date_filtered_mhdd_df = mhdd_df.filter(mhdd_df["Year"].isin([str(i) for i in range(1990, 2024)])) # Filter for 1990-2023
    
    # Transformation 2: Inspect and get rid of rows that have values > 100% for any of the percentage occurrences
    erroneous_vals_filtered_mhdd_df = date_filtered_mhdd_df.filter(~((date_filtered_mhdd_df["Schizophrenia (%)"]>100) | (date_filtered_mhdd_df["Bipolar disorder (%)"]>100) | (date_filtered_mhdd_df["Eating disorders (%)"]>100) | (date_filtered_mhdd_df["Drug use disorders (%)"]>100) | (date_filtered_mhdd_df["Depression (%)"]>100) | (date_filtered_mhdd_df["Alcohol use disorders (%)"]>100)))
    
    # Transformation 3: Select countries that are present in both datasets after unifying names across the board
    atlas_countries_merged = erroneous_vals_filtered_mhdd_df.join(atlas_countries_df.hint("broadcast"), (atlas_countries_df['country_code'] == erroneous_vals_filtered_mhdd_df['Code']))
    atlas_countries_merged = atlas_countries_merged.drop("Entity").drop("index").drop("Code")
    mhdd_df = atlas_countries_merged.withColumnsRenamed({
        "Year": "year",
        "Schizophrenia (%)": "schizophrenia_%",
        "Bipolar disorder (%)": "bipolar_disorder_%",
        "Eating disorders (%)": "eating_disorders_%",
        "Anxiety disorders (%)": "anxiety_disorders_%",
        "Drug use disorders (%)": "drug_use_disorders_%",
        "Depression (%)": "depression_%",
        "Alcohol use disorders (%)": "alcohol_use_disorders_%",
    }).select([
        "country_name",
        "country_code",
        "un_region",
        "year",
        "schizophrenia_%",
        "bipolar_disorder_%",
        "eating_disorders_%",
        "anxiety_disorders_%",
        "drug_use_disorders_%",
        "depression_%",
        "alcohol_use_disorders_%"
    ])
    save_output(mhdd_df, output_mhdd)
    
    # Prevalence of depression (male vs. female)
    dep_mf_df = spark.read.option("header", True).csv(input_dep_mf)
    
    dep_mf_df.cache() # Cache this because this is used a lot
    
    # Transformation 1: There are a lot of weird dates on the data. Some ranging all the way back to BCE. We want data only from 1990-2023 because it is more likely to be relevat and properly documented.
    date_filtered_dep_mf_df = dep_mf_df.filter(dep_mf_df["Year"].isin([str(i) for i in range(1990, 2024)])) # Filter for 1990-2023
    
    # Transformation 2: Rename columns
    date_filtered_dep_mf_df = date_filtered_dep_mf_df.withColumnsRenamed({
        "Prevalence - Depressive disorders - Sex: Male - Age: Age-standardized (Percent)": "male_depression_prevalence_%",
        "Prevalence - Depressive disorders - Sex: Female - Age: Age-standardized (Percent)": "female_depression_prevalence_%",
        "Population (historical estimates)": "Population"
    })
    
    # Transformation 3: Deal with null values
    null_filtered_dep_mf_df = date_filtered_dep_mf_df.filter(~((date_filtered_dep_mf_df["male_depression_prevalence_%"].isNull()) | (date_filtered_dep_mf_df["female_depression_prevalence_%"].isNull())))
    
    # Transformation 4: Merge with countries in Atlas Dataset
    final_dep_mf_df = null_filtered_dep_mf_df.join(atlas_countries_df.hint("broadcast"), (atlas_countries_df['country_code'] == null_filtered_dep_mf_df['Code']))
    final_dep_mf_df = final_dep_mf_df.drop("Entity").drop("index").drop("Code").drop("Continent").drop("Population")
    final_dep_mf_df = final_dep_mf_df.withColumnsRenamed({
        "Year": "year",
    }).select([
        "country_name",
        "country_code",
        "un_region",
        "year",
        "male_depression_prevalence_%",
        "female_depression_prevalence_%"
    ])
    save_output(final_dep_mf_df, output_dep_mf)
    
    # Prevalence of eating disorders (male vs. female)
    ed_mf_df = spark.read.option("header", True).csv(input_ed_mf)
    
    ed_mf_df.cache() # Cache this because this is used a lot
    
    # Transformation 1: There are a lot of weird dates on the data. Some ranging all the way back to BCE. We want data only from 1990-2023 because it is more likely to be relevat and properly documented.
    date_filtered_ed_mf_df = ed_mf_df.filter(ed_mf_df["Year"].isin([str(i) for i in range(1990, 2024)])) # Filter for 1990-2023
    
    # Transformation 2: Rename columns
    date_filtered_ed_mf_df = date_filtered_ed_mf_df.withColumnsRenamed({
        "Prevalence - Eating disorders - Sex: Male - Age: Age-standardized (Percent)": "male_eating_disorders_prevalence_%",
        "Prevalence - Eating disorders - Sex: Female - Age: Age-standardized (Percent)": "female_eating_disorders_prevalence_%",
        "Population (historical estimates)": "population"
    })
    
    # Transformation 3: Deal with null values
    null_filtered_ed_mf_df = date_filtered_ed_mf_df.filter(~((date_filtered_ed_mf_df["male_eating_disorders_prevalence_%"].isNull()) | (date_filtered_ed_mf_df["female_eating_disorders_prevalence_%"].isNull())))
    
    # Transformation 4: Merge with Atlas countries
    ed_final_df = null_filtered_ed_mf_df.join(atlas_countries_df.hint("broadcast"), (null_filtered_ed_mf_df["Code"] == atlas_countries_df["country_code"]))
    ed_final_df = ed_final_df.drop("index").drop("Code").drop("Entity").drop("Continent").drop("population").withColumnsRenamed({
        "Year": "year",
    }).select([
        "country_name",
        "country_code",
        "un_region",
        "year",
        "male_eating_disorders_prevalence_%",
        "female_eating_disorders_prevalence_%",
    ])
    save_output(ed_final_df, output_ed_mf)
    
    # Prevalence of anxiety disorders (male vs. female)
    ad_mf_df = spark.read.option("header", True).csv(input_ad_mf)
    
    ad_mf_df.cache() # Cache this because this is used a lot
    
    # Transformation 1: There are a lot of weird dates on the data. Some ranging all the way back to BCE. We want data only from 1990-2023 because it is more likely to be relevat and properly documented.
    date_filtered_ad_mf_df = ad_mf_df.filter(ad_mf_df["Year"].isin([str(i) for i in range(1990, 2024)])) # Filter for 1990-2023
    
    # Transformation 2: Rename columns
    date_filtered_ad_mf_df = date_filtered_ad_mf_df.withColumnsRenamed({
        "Prevalence - Anxiety disorders - Sex: Male - Age: Age-standardized (Percent)": "male_anxiety_disorders_prevalence_%",
        "Prevalence - Anxiety disorders - Sex: Female - Age: Age-standardized (Percent)": "female_anxiety_disorders_prevalence_%",
        "Population (historical estimates)": "population"
    })
    
    # Transformation 3: Deal with null values
    null_filtered_ad_mf_df = date_filtered_ad_mf_df.filter(~((date_filtered_ad_mf_df["male_anxiety_disorders_prevalence_%"].isNull()) | (date_filtered_ad_mf_df["female_anxiety_disorders_prevalence_%"].isNull())))
    
    # Transformation 4: Merge with Atlas countries
    ad_final_df = null_filtered_ad_mf_df.join(atlas_countries_df.hint("broadcast"), (null_filtered_ad_mf_df["Code"] == atlas_countries_df["country_code"]))
    ad_final_df = ad_final_df.drop("index").drop("Code").drop("Entity").drop("Continent").drop("population").withColumnsRenamed({
        "Year": "year",
    }).select([
        "country_name",
        "country_code",
        "un_region",
        "year",
        "male_anxiety_disorders_prevalence_%",
        "female_anxiety_disorders_prevalence_%",
    ])
    save_output(ad_final_df, output_ad_mf)
    
    # Prevalence of schizophrenia (male vs. female)
    sch_mf_df = spark.read.option("header", True).csv(input_sch_mf)
    
    sch_mf_df.cache() # Cache this because this is used a lot
    
    # Transformation 1: There are a lot of weird dates on the data. Some ranging all the way back to BCE. We want data only from 1990-2023 because it is more likely to be relevat and properly documented.
    date_filtered_sch_mf_df = sch_mf_df.filter(sch_mf_df["Year"].isin([str(i) for i in range(1990, 2024)])) # Filter for 1990-2023
    
    # Transformation 2: Rename columns
    date_filtered_sch_mf_df = date_filtered_sch_mf_df.withColumnsRenamed({
        "Prevalence - Schizophrenia - Sex: Male - Age: Age-standardized (Percent)": "male_schizophrenia_prevalence_%",
        "Prevalence - Schizophrenia - Sex: Female - Age: Age-standardized (Percent)": "female_schizophrenia_prevalence_%",
        "Population (historical estimates)": "population"
    })
    
    # Transformation 3: Deal with null values
    null_filtered_sch_mf_df = date_filtered_sch_mf_df.filter(~((date_filtered_sch_mf_df["male_schizophrenia_prevalence_%"].isNull()) | (date_filtered_sch_mf_df["female_schizophrenia_prevalence_%"].isNull())))
    
    # Transformation 4: Merge with Atlas countries
    sch_final_df = null_filtered_sch_mf_df.join(atlas_countries_df.hint("broadcast"), (null_filtered_sch_mf_df["Code"] == atlas_countries_df["country_code"]))
    sch_final_df = sch_final_df.drop("index").drop("Code").drop("Entity").drop("Continent").drop("population").withColumnsRenamed({
        "Year": "year",
    }).select([
        "country_name",
        "country_code",
        "un_region",
        "year",
        "male_schizophrenia_prevalence_%",
        "female_schizophrenia_prevalence_%",
    ])
    save_output(sch_final_df, output_sch_mf)
    
    # Prevalence of bipolar disorder (male vs. female)
    bpd_mf_df = spark.read.option("header", True).csv(input_bpd_mf)
    
    bpd_mf_df.cache() # Cache this because this is used a lot
    
    # Transformation 1: There are a lot of weird dates on the data. Some ranging all the way back to BCE. We want data only from 1990-2023 because it is more likely to be relevat and properly documented.
    date_filtered_bpd_mf_df = bpd_mf_df.filter(bpd_mf_df["Year"].isin([str(i) for i in range(1990, 2024)])) # Filter for 1990-2023
    
    # Transformation 2: Rename columns
    date_filtered_bpd_mf_df = date_filtered_bpd_mf_df.withColumnsRenamed({
        "Prevalence - Bipolar disorder - Sex: Male - Age: Age-standardized (Percent)": "male_bipolar_disorder_prevalence_%",
        "Prevalence - Bipolar disorder - Sex: Female - Age: Age-standardized (Percent)": "female_bipolar_disorder_prevalence_%",
        "Population (historical estimates)": "population"
    })
    
    # Transformation 3: Deal with null values
    null_filtered_bpd_mf_df = date_filtered_bpd_mf_df.filter(~((date_filtered_bpd_mf_df["male_bipolar_disorder_prevalence_%"].isNull()) | (date_filtered_bpd_mf_df["female_bipolar_disorder_prevalence_%"].isNull())))
    
    # Transformation 4: Merge with Atlas countries
    bpd_final_df = null_filtered_bpd_mf_df.join(atlas_countries_df.hint("broadcast"), (null_filtered_bpd_mf_df["Code"] == atlas_countries_df["country_code"]))
    bpd_final_df = bpd_final_df.drop("index").drop("Code").drop("Entity").drop("Continent").drop("population").withColumnsRenamed({
        "Year": "year",
    }).select([
        "country_name",
        "country_code",
        "un_region",
        "year",
        "male_bipolar_disorder_prevalence_%",
        "female_bipolar_disorder_prevalence_%",
    ])
    save_output(bpd_final_df, output_bpd_mf)
    
    # Suicide rate by country
    sui_2023_mf_df = spark.read.option("header", True).csv(input_sui_mf)
    
    sui_2023_mf_df.repartition(30) # There are 180 rows, work on 6 rows per partition
    sui_2023_mf_df.cache() # Cache this because this is used a lot
    
    # Transformation 1: Rename columns
    sui_2023_mf_df = sui_2023_mf_df.withColumnsRenamed({
        "rate2019both": "all_suicide_%",
        "rate2019male": "male_suicide_%",
        "rate2019female": "female_suicide_%"
    }).withColumn("year", F.lit("2019"))
    
    # Transformation 2: Remove erroneous values (percentage > 100%)
    erroneous_val_filtered_sui_2023_mf_df = sui_2023_mf_df.filter(~((sui_2023_mf_df["all_suicide_%"] > 100) | (sui_2023_mf_df["male_suicide_%"] > 100) | (sui_2023_mf_df["female_suicide_%"] > 100)))
    
    # Transformation 3: Add country codes
    country_dict_df_schema = types.StructType([
        types.StructField("country_dict_country", types.StringType()),
        types.StructField("country_dict_country_code", types.StringType()),
    ])
    country_dict_df = spark.read.csv(input_sui_countries_dict, schema=country_dict_df_schema)
    erroneous_val_filtered_sui_2023_mf_df = erroneous_val_filtered_sui_2023_mf_df.join(country_dict_df.hint("broadcast"), (erroneous_val_filtered_sui_2023_mf_df["country"]==country_dict_df["country_dict_country"]), how="left")
    erroneous_val_filtered_sui_2023_mf_df = erroneous_val_filtered_sui_2023_mf_df.drop("country").drop("country_dict_country").withColumnsRenamed({
        "country_dict_country_code": "code"
    })
    
    # Transformation 5: Merge with Atlas countries
    final_sui_2023_mf_df = erroneous_val_filtered_sui_2023_mf_df.join(atlas_countries_df.hint("broadcast"), (erroneous_val_filtered_sui_2023_mf_df["code"]==atlas_countries_df["country_code"]))
    final_sui_2023_mf_df = final_sui_2023_mf_df.drop("country").drop("code").select([
        "country_name",
        "country_code",
        "un_region",
        "year",
        "all_suicide_%",
        "male_suicide_%",
        "female_suicide_%"
    ])
    save_output(final_sui_2023_mf_df, output_sui_mf)