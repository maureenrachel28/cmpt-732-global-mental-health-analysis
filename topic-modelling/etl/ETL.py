#Import statements
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import countDistinct
import pandas as pd
from pyspark.sql.functions import year, month, count
from pyspark.sql.window import Window
from pyspark.sql.functions import col, to_date, year, month, date_format,sum,avg,corr,lit
from pyspark.sql.functions import lower, regexp_replace
from pyspark.sql.functions import explode, split, col, count
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.sql.functions import col, year, month, date_format, count
from pyspark.sql.functions import date_format,dayofweek



'''Creating spark session, loading data and cleaning dataset for analysis'''
#Create spark session object
spark = SparkSession.builder.appName("MH-Section3-ETL").getOrCreate()

path_to_csv_folder = "replace path"
df = spark.read.option("multiline", "true").option("quote", '"').option("header", "true").option("inferSchema", "true").option("sep", ",").option("escape", "\"").csv(path_to_csv_folder)

#Select columns required for analysis
selected_df = df.select("subreddit", "author", "date", "post").cache()

#Omit the ED/Anonymous subreddit because it does not span the timeframe that other subreddits do
remove_ED_filtered_df = selected_df.filter(col("subreddit") != "EDAnonymous").cache()

#Convert date column to to_date column
remove_ED_filtered_df = remove_ED_filtered_df.withColumn("date", to_date(col("date"), "yyyy/MM/dd"))

#Remove posts from Nov- Dec 2018, May - Dec 2019, to keep the timeframe even across the three years
timeframe_filtered_df = remove_ED_filtered_df.filter(
    ((col("month_year") >= "2018-01") & (col("month_year") <= "2018-04")) |
    ((col("month_year") >= "2019-01") & (col("month_year") <= "2019-04")) |
    ((col("month_year") >= "2020-01") & (col("month_year") <= "2020-04"))
).cache()


'''2. Increase in post activity pre and post covid in percentage
'''
# Count the number of posts per subreddit and month_year
posts_count = (
    timeframe_filtered_df
    .groupBy("subreddit", "month_year")
    .agg(count("post").alias("num_posts"))
)

# Calculate the average number of posts per subreddit before COVID
avg_posts_pre_covid = (
    posts_count
    .filter((col("month_year") >= "2018-01") & (col("month_year") < "2019-05"))
    .groupBy("subreddit")
    .agg(avg("num_posts").alias("avg_pre_covid"))
)

# Calculate the average number of posts per subreddit during COVID
avg_posts_covid = (
    posts_count
    .filter((col("month_year") >= "2020-01") & (col("month_year") <= "2020-04"))
    .groupBy("subreddit")
    .agg(avg("num_posts").alias("avg_during_covid"))
)

# Join the two dataframes on 'subreddit'
avg_combined_df = avg_posts_pre_covid.join(avg_posts_covid, "subreddit")

# Calculate the percentage increase for each subreddit
avg_combined_df = avg_combined_df.withColumn(
    "percentage_increase",
    ((col("avg_during_covid") - col("avg_pre_covid")) / col("avg_pre_covid")) * 100
)

#Convert to pandas
avg_combined_df_pandas = avg_combined_df.toPandas()

'''1. Calculating the suicidal ideation indicator word occurence in posts belonging to the subreddit suicidewatch'''
# Creating new df for this analysis 
indicators_df = remove_ED_filtered_df.select('post', 'subreddit', 'date')

#converting text in post to lower case
indicators_df = indicators_df.withColumn("post", lower(col("post")))

# Remove punctuation
indicators_df = indicators_df.withColumn("post", regexp_replace(col("post"), "[^\w\s]", ""))

#filter suicide subreddit
suicide_subreddit_df = indicators_df.filter(col("subreddit") == "suicidewatch")

#Add a month year column
suicide_subreddit_df = suicide_subreddit_df.withColumn("date", to_date(col("date"), "yyyy-MM-dd"))
suicide_subreddit_df = suicide_subreddit_df.withColumn("month_year", date_format(col("date"), "yyyy-MM"))


# Define the list of indicator words
suicidal_indicators = ["help", "done", "cant do this anymore", "end it", "no point", "tired", "worthless", "hopeless", "give up", "not want to live"]
# Tokenize the 'post' column into words
suicide_subreddit_df = suicide_subreddit_df.withColumn("words", split(col("post"), " "))

# Explode the 'words' column into individual words
suicide_subreddit_df = suicide_subreddit_df.withColumn("word", explode(col("words")))

# Filter the words to keep only the indicator words
suicide_subreddit_df = suicide_subreddit_df.where(col("word").isin(suicidal_indicators))

# Count the occurrences of each indicator word
word_count_df = suicide_subreddit_df.groupBy("word").agg(count("word").alias("counts"))

#Convert to pandas
word_count_df_pandas = word_count_df.toPandas()

#Save dataframe to csv
word_count_df_pandas.to_csv("/content/suicde_indicators_per_subreddit.csv")

'''3. Word Cloud on subreddit anxiety'''
anxiety_subreddit_df = remove_ED_filtered_df.filter(col("subreddit") == "anxiety")

#Converting post column to lowercase
anxiety_subreddit_df = anxiety_subreddit_df.withColumn("post", lower(col("post")))

#removing punctuation
anxiety_subreddit_df = anxiety_subreddit_df.withColumn("post", regexp_replace(col("post"), "[^\w\s]", ""))

#Adding column words
anxiety_subreddit_df = anxiety_subreddit_df.withColumn("words", split(col("post"), " "))

#Removing stop words
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
filtered_df = remover.transform(anxiety_subreddit_df)

words_df = anxiety_subreddit_df.select(explode(col("words")).alias("word"))
words_list = words_df.rdd.flatMap(lambda x: x).collect()
words_list_df = pd.DataFrame(words_list, columns=["Words"])

#save the dataframe as csv
words_list_df.to_csv('/content/anxiety_wordcloud_df.csv')


'''4. User Engagement Analysis'''

#User engagement analysis



# Filter posts from Jan 2019 to Dec 2019
filtered_df = remove_ED_filtered_df.filter(
    (year(col("date")) == 2019) &
    (month(col("date")) >= 1) &
    (month(col("date")) <= 12)
)

# Post count per month
posts_per_month = filtered_df.groupBy(date_format(col("date"), "yyyy-MM").alias("month")).count().orderBy("month")

# Post count by author
posts_by_author = filtered_df.groupBy("author").count().orderBy(col("count").desc())

# Convert to Pandas DataFrame for visualization
posts_per_month_pd = posts_per_month.toPandas()
posts_by_author_pd = posts_by_author.toPandas()

# Group by subreddit and month
posts_per_month_subreddit = filtered_df.groupBy(
    "subreddit",
    date_format(col("date"), "yyyy-MM").alias("month")
).count().orderBy("subreddit", "month")

# Convert to Pandas DataFrame
posts_per_month_subreddit_pd = posts_per_month_subreddit.toPandas()

posts_per_month_subreddit_pd.to_csv("/content/posts_per_subreddit.csv")


# Group by subreddit
posts_by_subreddit = filtered_df.groupBy("subreddit").count().orderBy(col("count").desc())

# Convert to Pandas DataFrame
posts_by_subreddit_pd = posts_by_subreddit.toPandas()

posts_by_subreddit_pd.to_csv("/content/distribution_subreddit.csv")

# Add a column for the day of the week
remove_ED_filtered_df = remove_ED_filtered_df.withColumn("day_of_week", dayofweek(col("date")))

# Group by subreddit and day of the week
posts_by_day_subreddit = remove_ED_filtered_df.groupBy("subreddit", "day_of_week").count()

# Convert to Pandas DataFrame for visualization
posts_by_day_subreddit_pd = posts_by_day_subreddit.toPandas()

posts_by_day_subreddit_pd.to_csv("/content/weekly_posts.csv")

