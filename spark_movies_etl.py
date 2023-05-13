from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, split, col

# Create a SparkSession
spark = SparkSession.builder.appName("MovieETL").getOrCreate()

# Load the movies dataframe from the movies.csv file
movies_df = spark.read.csv("./ml-latest-small/movies.csv", header=True, inferSchema=True)

# Load the ratings dataframe from the ratings.csv file
ratings_df = spark.read.csv("ml-latest-small/ratings.csv", header=True, inferSchema=True)

# Load the links dataframe from the links.csv file
links_df = spark.read.csv("ml-latest-small/links.csv", header=True, inferSchema=True)

# Join the movies and ratings dataframes on the movieId column
movies_ratings_df = movies_df.join(ratings_df, "movieId")

# Join the movies_ratings dataframe and the links dataframe on the movieId column
movies_ratings_links_df = movies_ratings_df.join(links_df, "movieId")

# Split the title column on the first parentheses to extract the year information
movies_ratings_links_df = movies_ratings_links_df.withColumn("year", split(col("title"), "\(").getItem(1))
movies_ratings_links_df = movies_ratings_links_df.withColumn("year", split(col("year"), "\)").getItem(0))

# Convert the year column to an integer type
movies_ratings_links_df = movies_ratings_links_df.withColumn("year", movies_ratings_links_df["year"].cast("integer"))



# Filter the DataFrame to include only movies released after 2000
movies_ratings_links_df = movies_ratings_links_df.filter(movies_ratings_links_df["year"] > 2000)

# Calculate the average rating for each genre
avg_ratings = movies_ratings_links_df.groupBy("genres").agg(avg("rating").alias("avg_rating"))

# Print the schema and the DataFrame
avg_ratings.printSchema()
avg_ratings.show(100)

# Write the results to a Parquet file
avg_ratings.write.mode("overwrite").parquet("avg_ratings.parquet")

# Stop the SparkSession
spark.stop()