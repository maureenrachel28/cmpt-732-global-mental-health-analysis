from pyspark.sql import functions as F
from pyspark.sql import Row, types

# This is a dependency
# from countryguess import guess_country
import re

# @F.udf(returnType=types.StringType())
# def get_country_code(country):
#     return guess_country(country)["iso3"]

# @F.udf(returnType=types.StringType())
# def get_country_name(country):
#     return guess_country(country)["name_official"]

# @F.udf(returnType=types.StringType())
# def get_country_region(country):
#     return guess_country(country)["unregion"]

@F.udf(returnType=types.FloatType())
def convert_to_score(value, _array):
    if value > _array[0][8]: # higher than the 90th-percentile
        return 0.99
    elif value > _array[0][7]:
        return 0.88
    elif value > _array[0][6]:
        return 0.77
    elif value > _array[0][5]:
        return 0.66
    elif value > _array[0][4]:
        return 0.55
    elif value > _array[0][3]:
        return 0.44
    elif value > _array[0][2]:
        return 0.33
    elif value > _array[0][1]:
        return 0.22
    elif value > _array[0][0]:
        return 0.11
    else:
        return 0.0
    
@F.udf(returnType=types.IntegerType())
def convert_to_binary(value):
    if value is None:
        return 0
    else:
        return 1

@F.udf(returnType=types.IntegerType())
def convert_population(population):
    if population == "-":
        return None
    if isinstance(population, type(None)):
        return
    if isinstance(population, int):
        return population
    population = population.replace(" ", "").replace(",", "").replace(", ", "")
    return int(population)

@F.udf(returnType=types.StringType())
def convert_who_region(region):
    if region == "-":
        return None
    if isinstance(region, type(None)):
        return
    return region.lower()

@F.udf(returnType=types.StringType())
def convert_income_group(val):
    if val == "-":
        return None
    if isinstance(val, type(None)):
        return
    return val.lower()

code_pattern = r'[A-Z]{3}'

@F.udf(returnType=types.StringType())
def extract_code(value):
    if value == "-":
        return None
    if isinstance(value, type(None)):
        return
    value_str = str(value).upper()
    if re.search(code_pattern, value_str):
        match = re.search(code_pattern, value_str)
        if match:
            start, end = match.span()
            return value_str[start:end] 
        else:
            None

@F.udf(returnType=types.FloatType())
def extract_exp(value):
    if value == "-":
        return None
    if isinstance(value, type(None)):
        return
    value_str = str(value).upper()
    if re.search(code_pattern, value_str):
        match = re.search(code_pattern, value_str)
        if match:
            start, end = match.span()
            return float(value_str[0:start-1].replace(" ", "").replace(",", "").replace(", ", "")) 
        else:
            None
            
@F.udf(returnType=types.FloatType())
def convert_smr(value):
    if value == "-":
        return None
    if isinstance(value, type(None)):
        return
    else:
        return float(value)
    
@F.udf(returnType=types.StringType())
def convert_dash_to_null_string(value):
    if value == "-":
        return None
    elif isinstance(value, type(None)):
        return
    else:
        return str(value)

# N - does not exist, EW - exists and functions well, EP - exists and functions poorly
@F.udf(returnType=types.StringType())
def convert_authority(value):
    transformation_dictionary = {
        'A dedicated body authority does not exist': "N",
        'A dedicated authority body does  not exist': "N",
        'Exist and provides regular inspections of facilities and reports at least annually': "EW",
        'A dedicated body authority undertakes regular inspections, responds to complaints, and reports its findings at least once a year': "EW",
        'A dedicated authority body does not exist': "N",
        'A dedicated authority undertakes regular inspections,responds to complaints and reports its findings at least once a year': "EW",
        'A dedicated body exists but it is not functioning well': "EP",
        'exists but not fully functioning': "EP",
        'A dedicated authority undertakes  irregular inspections of mental  health services and irregularly  responds to complaints of human  rights violations': "EP",
        'A dedicated body does not exist': "N",
        'A dedicated body undertakes irregular inspections of mental health services and irregularly responds to complaints of human rights violations': "EP",
        'A dedicated authority undertakes irregular inspections of mental health services and irregularly responds to complaints of human rights violations': "EP",
        'A dedicated authority body exists  but it is not functioning well': "EP",
        'A dedicated authority undertakes regular inspections, responds to complaints, and reports its finding at least once a year': "EW",
        'A dedicated authority undertakes regular inspections, responds to complaints, and reports its findings at least once a year': "EW",
        'A dedicated authority body exists but it is not functioning well': "EP",
        'A dedicated authority undertakes regular inspections, responds to complaints and reports its findings at least once a year': "EW",
        'A dedicated authority undertakes  regular inspections, responds to  complaints, and reports its findings  at least once a year': "EW"
    }
    if value == "-":
        return None
    if isinstance(value, type(None)) or value in ["N", "EW", "EP"]:
        return value
    else:
        return transformation_dictionary[value]
    
@F.udf(returnType=types.FloatType())
def convert_dash_to_null_float(value):
    if value == "-" or value == "--":
        return None
    elif isinstance(value, type(None)):
        return
    elif isinstance(value, float):
        return value
    else:
        return float(value.replace(" ", "").replace(",", "").replace(", ", ""))
    
# FI - fully insured, 20 - at least 20% paid by individual, M - mostly paid by individual
@F.udf(returnType=types.StringType())
def convert_payment(value):
    transformation_dictionary = {
        'Persons pay nothing at the point of service use (fully insured)': "FI",
        'Persons pay mostly or entirely out of pocket for services': "M",
        'Persons pay mostly or entirely out of pocket medicines': "M",
        'Persons pay mostly or entirely out of pocket for medicines': "M",
        'Persons pay at least 20% towards the cost of services': "20",
        'Persons pay at least 20% towards the cost of medicines': "20",
        'Persons pay atleast 20% towards the cost of services': "20",
        'nothing, fully insured': "FI",
        'at least 20% paid by individual': "20",
        'Persons pay nothing at the point of service use(fully insured)': "FI",
        'Personds pay nothing at the point of service use(fully insured)': "FI",
        'persons pay nothing at the point of service use(fully insured)': "FI",
        'Personds pay nothing at the point of service use (fully insured)': "FI",
        'mostly paid by individuals': "M"
    }
    if value == "-":
        return None
    if isinstance(value, type(None)) or value in ["FI", "20", "M"]:
        return value
    else:
        return transformation_dictionary[value]
    
@F.udf(returnType=types.FloatType())
def convert_percentage_to_category_float(value):
    transformation_dictionary = {
        "less-25": 1,
        "25-less": 1,
        "25": 1,
        "26-50": 2,
        "51-75": 3,
        "more-75": 4,
        "More-75": 4
    }
    if value == "-" or value == "--":
        return None
    elif isinstance(value, type(None)):
        return
    elif isinstance(value, float):
        return value
    else:
        return float(transformation_dictionary[value])

@F.udf(returnType=types.StringType())
def convert_dash_to_null_table_string(value):
    if value == "-" or value == "-,-,-,-" or value == "??":
        return None
    elif isinstance(value, type(None)):
        return
    else:
        return value