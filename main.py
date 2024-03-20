import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

from pyspark.sql import SparkSession, types
from pyspark.sql import functions as F

from etl_scripts import atlas_countries_fact_dim, global_mental_health
from analysis_scripts import gmh_analysis, atlas_gmh_analysis

import time

if __name__ == '__main__':
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    spark = SparkSession.builder.appName('BDL Project - Team ML').getOrCreate()
    assert spark.version >= '3.0' # make sure we have Spark 3.0+
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    start = time.time()
    atlas_countries_fact_dim.run(input_dir, output_dir)
    ckpt1 = time.time()
    print("Creating facts and dimensions from Atlas Dataset took {duration} seconds".format(duration=ckpt1-start))
    global_mental_health.run(input_dir, output_dir)
    ckpt2 = time.time()
    print("Creating facts and dimensions from GMH Dataset took {duration} seconds".format(duration=ckpt2-ckpt1))
    gmh_analysis.run()
    ckpt3 = time.time()
    print("Running analysis script on GMH Dataset took {duration} seconds".format(duration=ckpt3-ckpt2))
    atlas_gmh_analysis.run()
    ckpt4 = time.time()
    print("Running analysis script on Atlas+GMH Dataset took {duration} seconds".format(duration=ckpt4-ckpt3))
    