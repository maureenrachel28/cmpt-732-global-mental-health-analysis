import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

from pyspark.sql import SparkSession, types
from pyspark.sql import functions as F

spark = SparkSession.builder.appName('BDL Project - Team ML').getOrCreate()
assert spark.version >= '3.0' # make sure we have Spark 3.0+
spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext

# Helper function
def save_pandas_csv(df, location, filename):
    df.to_csv(location + "/" + filename + '.csv', index=False)
    
# Analysis functions
def generate_top_bottom_mean_for_problem_in_mhdd(mhdd_df, problem):
    # Select year, country_name, problem and add average as row
    problem_average = mhdd_df.groupby("year").agg(F.avg(problem).alias(problem)).withColumn("country_name", F.lit("Mean")).select(["year", "country_name", problem])

    problem_df = mhdd_df.select(["year", "country_name", problem])

    problem_df_final = problem_df.union(problem_average)

    # Find top 5 and bottom 5 countries
    top_bottom = mhdd_df.groupBy("country_name").agg(F.avg(problem).alias(problem)).sort(problem, ascending=False)
    top_5_countries = top_bottom.select(["country_name"]).limit(5).collect()
    countries_list = [top_5_countries[i]["country_name"] for i in range(len(top_5_countries))]
    bottom_5_countries = top_bottom.select(["country_name"]).tail(5)
    countries_list = countries_list + [bottom_5_countries[i]["country_name"] for i in range(len(bottom_5_countries))]
    countries_list.append("Mean")
    return problem_df_final.select(["year", "country_name", problem]).filter((mhdd_df["country_name"].isin(countries_list))).toPandas()

def problematic_region_df(combined_mhdd_df, group, problem):
    group_val = combined_mhdd_df.groupby(group).agg(F.avg(problem).alias(problem)).orderBy(F.desc(problem)).first()[group]
    viz_df = combined_mhdd_df.filter(combined_mhdd_df[group]==group_val).groupby("country_name").agg(F.avg(problem).alias(problem))
    return (group_val, viz_df.toPandas())

def generate_problem_growth_df(combined_mhdd_df, problem):
    test = combined_mhdd_df.select(["un_region", "year", problem]).groupby("un_region", "year").agg(
        F.avg(problem).alias(problem),
    ).filter((F.col("year")==1990)|(F.col("year")==2017))

    test_1990 = test.filter(test["year"]==1990).drop("year").withColumnsRenamed({
        problem: problem + "_1990",
        "un_region": "un_region_1990"
    })

    test_2017 = test.filter(test["year"]==2017).drop("year").withColumnsRenamed({
        problem: problem + "_2017",
        "un_region": "un_region_2017"
    })

    diff_df = test_1990.join(test_2017, (test_1990["un_region_1990"]==test_2017["un_region_2017"])).drop("un_region_2017").withColumnRenamed("un_region_1990", "un_region")
    diff_df = diff_df.withColumn(problem.capitalize()+" growth", F.col(problem+"_2017")-F.col(problem+"_1990")).drop(problem+"_1990").drop(problem+"_2017")

    return diff_df.toPandas()

def generate_correlation_heatmap_df(combined_mhdd_df):
    averaged_df = combined_mhdd_df.groupby("un_region", "year").agg(F.avg(F.col("schizophrenia_%")).alias("Schizophrenia"), F.avg(F.col("depression_%")).alias("Depression"), F.avg(F.col("bipolar_disorder_%")).alias("Bipolar disorder"), F.avg(F.col("eating_disorders_%")).alias("Eating disorders"), F.avg(F.col("anxiety_disorders_%")).alias("Anxiety disorders"), F.avg(F.col("drug_use_disorders_%")).alias("Drug use disorders"), F.avg(F.col("alcohol_use_disorders_%")).alias("Alcohol use disorders"))
    return averaged_df.select(["Schizophrenia", "Depression", "Drug use disorders", "Alcohol use disorders", "Eating disorders", "Anxiety disorders", "Bipolar disorder"]).toPandas()

# Main function
def run():

    # MHDD schema
    mhdd_schema = types.StructType([
        types.StructField("country_name", types.StringType()),
        types.StructField("country_code", types.StringType()),
        types.StructField("un_region", types.StringType()),
        types.StructField("year", types.IntegerType()),
        types.StructField("schizophrenia_%", types.FloatType()),
        types.StructField("bipolar_disorder_%", types.FloatType()),
        types.StructField("eating_disorders_%", types.FloatType()),
        types.StructField("anxiety_disorders_%", types.FloatType()),
        types.StructField("drug_use_disorders_%", types.FloatType()),
        types.StructField("depression_%", types.FloatType()),
        types.StructField("alcohol_use_disorders_%", types.FloatType()),
    ])
    
    # Load MHDD dataset
    mhdd_df = spark.read.option("multiline", "true").csv("datasets_output/gmh_fact_dims/mhdd", schema=mhdd_schema)
    mhdd_df.cache()
    
    mhdd_viz_df = mhdd_df.toPandas()
    
    save_pandas_csv(mhdd_viz_df, "csv_out/gmh", "mhdd_viz_df")
    
    # Generate the top 5, mean and bottom 5 for all problems
    # Not caching these since they don't get used in other operations
    sch_df = generate_top_bottom_mean_for_problem_in_mhdd(mhdd_df, "schizophrenia_%")
    bpd_df = generate_top_bottom_mean_for_problem_in_mhdd(mhdd_df, "bipolar_disorder_%")
    ed_df = generate_top_bottom_mean_for_problem_in_mhdd(mhdd_df, "eating_disorders_%")
    ad_df = generate_top_bottom_mean_for_problem_in_mhdd(mhdd_df, "anxiety_disorders_%")
    du_df = generate_top_bottom_mean_for_problem_in_mhdd(mhdd_df, "drug_use_disorders_%")
    au_df = generate_top_bottom_mean_for_problem_in_mhdd(mhdd_df, "alcohol_use_disorders_%")
    dep_df = generate_top_bottom_mean_for_problem_in_mhdd(mhdd_df, "depression_%")
    
    save_pandas_csv(sch_df, "csv_out/gmh", "sch_df")
    save_pandas_csv(bpd_df, "csv_out/gmh", "bpd_df")
    save_pandas_csv(ed_df, "csv_out/gmh", "ed_df")
    save_pandas_csv(ad_df, "csv_out/gmh", "ad_df")
    save_pandas_csv(du_df, "csv_out/gmh", "du_df")
    save_pandas_csv(au_df, "csv_out/gmh", "au_df")
    save_pandas_csv(dep_df, "csv_out/gmh", "dep_df")
    
    # Atlas countries facts schema
    atlas_countries_facts_schema = types.StructType([
        types.StructField("country_name", types.StringType()),
        types.StructField("facts_country_code", types.StringType()),
        types.StructField("un_region", types.StringType()),
    ])

    atlas_countries_facts = spark.read.csv("datasets_output/atlas_fact_dims/atlas_countries_facts/", schema=atlas_countries_facts_schema)
    
    # Atlas countries basic info schema
    atlas_countries_basic_info_schema = types.StructType([
        types.StructField("basic_country_code", types.StringType()),
        types.StructField("population", types.IntegerType()),
        types.StructField("income_group", types.StringType()),
        types.StructField("who_region", types.StringType()),
        types.StructField("expenditure_cad", types.FloatType()),
    ])

    atlas_countries_basic_info = spark.read.csv("datasets_output/atlas_fact_dims/atlas_countries_basic_info_dims/", schema=atlas_countries_basic_info_schema)
    
    # Join basic info and facts
    atlas_basic_combined = atlas_countries_basic_info.join(atlas_countries_facts.hint("broadcast"), (atlas_countries_basic_info["basic_country_code"]==atlas_countries_facts["facts_country_code"]))
    
    # Drop some unnecessary columns
    atlas_basic_combined = atlas_basic_combined.drop("facts_country_code").drop("un_region").drop("who_region").drop("country_name")
    
    # Combine MHDD with basic combined
    combined_mhdd_df = mhdd_df.join(atlas_basic_combined.hint("broadcast"), (mhdd_df["country_code"]==atlas_basic_combined["basic_country_code"])).drop("basic_country_code")
    combined_mhdd_df.cache() # Cache this since this will be used in a lot of operations
    
    # Find region with highest problem prevalence (taking average across un_region or income_group) for each problem
    # Not caching these since they don't get used in other operations
    sch_un_group, sch_df_un_region = problematic_region_df(combined_mhdd_df, "un_region", "schizophrenia_%")
    sch_inc_group, sch_df_income_group = problematic_region_df(combined_mhdd_df, "income_group", "schizophrenia_%")
    bpd_un_group, bpd_df_un_region = problematic_region_df(combined_mhdd_df, "un_region", "bipolar_disorder_%")
    bpd_inc_group, bpd_df_income_group = problematic_region_df(combined_mhdd_df, "income_group", "bipolar_disorder_%")
    ed_un_group, ed_df_un_region = problematic_region_df(combined_mhdd_df, "un_region", "eating_disorders_%")
    ed_inc_group, ed_df_income_group = problematic_region_df(combined_mhdd_df, "income_group", "eating_disorders_%")
    ad_un_group, ad_df_un_region = problematic_region_df(combined_mhdd_df, "un_region", "anxiety_disorders_%")
    ad_inc_group, ad_df_income_group = problematic_region_df(combined_mhdd_df, "income_group", "anxiety_disorders_%")
    du_un_group, du_df_un_region = problematic_region_df(combined_mhdd_df, "un_region", "drug_use_disorders_%")
    du_inc_group, du_df_income_group = problematic_region_df(combined_mhdd_df, "income_group", "drug_use_disorders_%")
    dep_un_group, dep_df_un_region = problematic_region_df(combined_mhdd_df, "un_region", "depression_%")
    dep_inc_group, dep_df_income_group = problematic_region_df(combined_mhdd_df, "income_group", "depression_%")
    au_un_group, aud_df_un_region = problematic_region_df(combined_mhdd_df, "un_region", "alcohol_use_disorders_%")
    au_inc_group, aud_df_income_group = problematic_region_df(combined_mhdd_df, "income_group", "alcohol_use_disorders_%")
    
    save_pandas_csv(sch_df_un_region, "csv_out/gmh", "sch_df_un_region")
    save_pandas_csv(bpd_df_un_region, "csv_out/gmh", "bpd_df_un_region")
    save_pandas_csv(ed_df_un_region, "csv_out/gmh", "ed_df_un_region")
    save_pandas_csv(ad_df_un_region, "csv_out/gmh", "ad_df_un_region")
    save_pandas_csv(du_df_un_region, "csv_out/gmh", "du_df_un_region")
    save_pandas_csv(aud_df_un_region, "csv_out/gmh", "au_df_un_region")
    save_pandas_csv(dep_df_un_region, "csv_out/gmh", "dep_df_un_region")
    save_pandas_csv(sch_df_income_group, "csv_out/gmh", "sch_df_income_group")
    save_pandas_csv(bpd_df_income_group, "csv_out/gmh", "bpd_df_income_group")
    save_pandas_csv(ed_df_income_group, "csv_out/gmh", "ed_df_income_group")
    save_pandas_csv(ad_df_income_group, "csv_out/gmh", "ad_df_income_group")
    save_pandas_csv(du_df_income_group, "csv_out/gmh", "du_df_income_group")
    save_pandas_csv(aud_df_income_group, "csv_out/gmh", "au_df_income_group")
    save_pandas_csv(dep_df_income_group, "csv_out/gmh", "dep_df_income_group")
    
    # Find average (across un_region) growth from 1990-2017 for each problem
    # Not caching these since they don't get used in other operations
    sch_problem_growth_viz_df = generate_problem_growth_df(combined_mhdd_df, "schizophrenia_%")
    ad_problem_growth_viz_df = generate_problem_growth_df(combined_mhdd_df, "anxiety_disorders_%")
    au_problem_growth_viz_df = generate_problem_growth_df(combined_mhdd_df, "alcohol_use_disorders_%")
    ed_problem_growth_viz_df = generate_problem_growth_df(combined_mhdd_df, "eating_disorders_%")
    du_problem_growth_viz_df = generate_problem_growth_df(combined_mhdd_df, "drug_use_disorders_%")
    bpd_problem_growth_viz_df = generate_problem_growth_df(combined_mhdd_df, "bipolar_disorder_%")
    dep_problem_growth_viz_df = generate_problem_growth_df(combined_mhdd_df, "depression_%")
    
    save_pandas_csv(sch_problem_growth_viz_df, "csv_out/gmh", "sch_problem_growth_viz_df")
    save_pandas_csv(ad_problem_growth_viz_df, "csv_out/gmh", "ad_problem_growth_viz_df")
    save_pandas_csv(au_problem_growth_viz_df, "csv_out/gmh", "au_problem_growth_viz_df")
    save_pandas_csv(ed_problem_growth_viz_df, "csv_out/gmh", "ed_problem_growth_viz_df")
    save_pandas_csv(du_problem_growth_viz_df, "csv_out/gmh", "du_problem_growth_viz_df")
    save_pandas_csv(bpd_problem_growth_viz_df, "csv_out/gmh", "bpd_problem_growth_viz_df")
    save_pandas_csv(dep_problem_growth_viz_df, "csv_out/gmh", "dep_problem_growth_viz_df")
    
    # Generate a correlation DF by taking average across all un_regions and years for each problem
    # Running df.corr() here
    correlation_viz_df = generate_correlation_heatmap_df(combined_mhdd_df)
    
    save_pandas_csv(correlation_viz_df, "csv_out/gmh", "correlation_viz_df")
    
    # Load ad_mf_df dataset
    ad_schema = types.StructType([
        types.StructField("country_name", types.StringType()),
        types.StructField("country_code", types.StringType()),
        types.StructField("un_region", types.StringType()),
        types.StructField("year", types.IntegerType()),
        types.StructField("male_anxiety_disorders_%", types.FloatType()),
        types.StructField("female_anxiety_disorders_%", types.FloatType()),
    ])
    ad_mf_df = spark.read.option("multiline", "true").csv("datasets_output/gmh_fact_dims/ad_mf", schema=ad_schema)
    
    # Load ad_mf_df dataset
    bpd_schema = types.StructType([
        types.StructField("country_name", types.StringType()),
        types.StructField("country_code", types.StringType()),
        types.StructField("un_region", types.StringType()),
        types.StructField("year", types.IntegerType()),
        types.StructField("male_bipolar_disorder_%", types.FloatType()),
        types.StructField("female_bipolar_disorder_%", types.FloatType()),
    ])
    bpd_mf_df = spark.read.option("multiline", "true").csv("datasets_output/gmh_fact_dims/bpd_mf", schema=bpd_schema)
    
    # Load ad_mf_df dataset
    dep_schema = types.StructType([
        types.StructField("country_name", types.StringType()),
        types.StructField("country_code", types.StringType()),
        types.StructField("un_region", types.StringType()),
        types.StructField("year", types.IntegerType()),
        types.StructField("male_depression_%", types.FloatType()),
        types.StructField("female_depression_%", types.FloatType()),
    ])
    dep_mf_df = spark.read.option("multiline", "true").csv("datasets_output/gmh_fact_dims/dep_mf", schema=dep_schema)
    
    # Load ad_mf_df dataset
    sch_schema = types.StructType([
        types.StructField("country_name", types.StringType()),
        types.StructField("country_code", types.StringType()),
        types.StructField("un_region", types.StringType()),
        types.StructField("year", types.IntegerType()),
        types.StructField("male_schizophrenia_%", types.FloatType()),
        types.StructField("female_schizophrenia_%", types.FloatType()),
    ])
    sch_mf_df = spark.read.option("multiline", "true").csv("datasets_output/gmh_fact_dims/sch_mf", schema=sch_schema)
    
    # Load ad_mf_df dataset
    ed_schema = types.StructType([
        types.StructField("country_name", types.StringType()),
        types.StructField("country_code", types.StringType()),
        types.StructField("un_region", types.StringType()),
        types.StructField("year", types.IntegerType()),
        types.StructField("male_eating_disorders_%", types.FloatType()),
        types.StructField("female_eating_disorders_%", types.FloatType()),
    ])
    ed_mf_df = spark.read.option("multiline", "true").csv("datasets_output/gmh_fact_dims/ed_mf", schema=ed_schema)
    
    ad_mf_df_pd = ad_mf_df.toPandas()
    bpd_mf_df_pd = bpd_mf_df.toPandas()
    dep_mf_df_pd = dep_mf_df.toPandas()
    sch_mf_df_pd = sch_mf_df.toPandas()
    ed_mf_df_pd = ed_mf_df.toPandas()
    
    save_pandas_csv(ad_mf_df_pd, "csv_out/gmh", "ad_mf_df_pd")
    save_pandas_csv(bpd_mf_df_pd, "csv_out/gmh", "bpd_mf_df_pd")
    save_pandas_csv(dep_mf_df_pd, "csv_out/gmh", "dep_mf_df_pd")
    save_pandas_csv(sch_mf_df_pd, "csv_out/gmh", "sch_mf_df_pd")
    save_pandas_csv(ed_mf_df_pd, "csv_out/gmh", "ed_mf_df_pd")
    
    # Load ad_mf_df dataset
    sui_schema = types.StructType([
        types.StructField("country_name", types.StringType()),
        types.StructField("country_code", types.StringType()),
        types.StructField("un_region", types.StringType()),
        types.StructField("year", types.IntegerType()),
        types.StructField("male_suicide_%", types.FloatType()),
        types.StructField("female_suicide_%", types.FloatType()),
    ])
    sui_mf_df = spark.read.option("multiline", "true").csv("datasets_output/gmh_fact_dims/sui_mf", schema=sui_schema)
    
    # Find average suicide % by region
    region_avg_sui_mf_df = sui_mf_df.select(["un_region", "year", "male_suicide_%", "female_suicide_%"]).groupby("un_region", "year").agg(F.avg("male_suicide_%").alias("male_suicide_%"), F.avg("female_suicide_%").alias("female_suicide_%"))
    
    # Find top 5 and bottom 5 by male and female
    top_male_region_sui_df = region_avg_sui_mf_df.orderBy(F.desc("male_suicide_%")).limit(5)
    top_female_region_sui_df = region_avg_sui_mf_df.orderBy(F.desc("female_suicide_%")).limit(5)
    bottom_male_region_sui_df = region_avg_sui_mf_df.orderBy("male_suicide_%").limit(5)
    bottom_female_region_sui_df = region_avg_sui_mf_df.orderBy("female_suicide_%").limit(5)

    # Union and then drop duplicates
    combined_sui_df = top_male_region_sui_df.union(top_female_region_sui_df).union(bottom_male_region_sui_df).union(bottom_female_region_sui_df).drop_duplicates()
    
    combined_sui_df_pd = combined_sui_df.toPandas()
    save_pandas_csv(combined_sui_df_pd, "csv_out/gmh", "combined_sui_df_pd")
    
    
    
    