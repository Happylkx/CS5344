# Algorithm:
# Step 1. Find the number of reviews each day and calculate the average ratings of the
# reviews from the review file. Use pair RDD, reduceByKey and map function to
# accomplish this step. The key is a tuple (the product ID/asin, review time). The
# value is a tuple (#review, average_ratings).
# Step 2. Create an RDD where the key is the product ID/asin and value is the brand
# name of the product. Use the metadata for this step.
# Step 3. Join the pair RDD obtained in Step 1 and the RDD created in Step 2.
# Step 4. Find the top 15 products with the greatest number of views in a day.
# Step 5. Output the number of reviews, review time, average ratings and product brand
# name for the top 15 products identified in Step 4.
# Input: Review file and metadata.
# Output: One line per product in the following format:
# <product ID > <num of reviews> <review time> <avg ratings> <product brand name>
import json
import pyspark as spark
from pyspark import SparkContext, SparkConf

def parse_reviews(line):
    item = json.loads(line)
    return ((item["asin"], item["reviewTime"]), (1, item["overall"]))

def parse_metas(line):
    item = json.loads(line)
    return (item["asin"], item["brand"])
    
conf = SparkConf().setAppName("Products")
sc = SparkContext(conf=conf)

# Step 1: Find number of reviews and average ratings
reviews_file = sc.textFile("Patio_Lawn_and_Garden.json")
metas_file = sc.textFile("meta_Patio_Lawn_and_Garden.json")


# Convert the review rating to float and the review count to integer
# reviewTime field is already day date, no need for conversion
reviews_pair_rdd = reviews_file.map(parse_reviews)
# Add count and overall score
# key is automatically handled: (asin, reviewTime)
# x, y are values: (count, overall)
reviews_count_avg = reviews_pair_rdd.reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
# x: ((asin, reviewTime), (sum_count, sum_overall))
reviews_avg_rdd = reviews_count_avg.map(lambda x: (x[0], (x[1][0], x[1][1] / x[1][0])))


# Step 2: Create RDD with product ID and brand name
# (asin, brand)
product_brand_rdd = metas_file.map(parse_metas)

# Step 3: Join the RDDs
# (asin, (reviewTime, sum_count, avg_overall_rating))
temp = reviews_avg_rdd.map(lambda x: (x[0][0], (x[0][1], x[1][0], x[1][1])))

# (asin, ((reviewTime, sum_count, avg_overall_rating), brand))
joined_rdd = temp.join(product_brand_rdd)

# Revert format changes:
# ((asin, brand, reviewTime), (sum_count, avg_overall_rating))
joined_rdd = joined_rdd.map(lambda x: ((x[0], x[1][1], x[1][0][0]), (x[1][0][1], x[1][0][2])))


# Step 4: Find top 15 products with greatest number of views
top_products = joined_rdd.sortBy(lambda x: x[1][0], ascending=False).take(15)


# Step 5: Output the results
with open("out.txt", 'w') as f:
	for ((asin, brand, review_time), (sum_count, avg_overall_rating)) in top_products:
	    f.write(f"{asin} {sum_count} {review_time} {avg_overall_rating} {brand}\n")

# Stop Spark
sc.stop()
