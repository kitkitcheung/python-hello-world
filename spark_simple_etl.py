from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, when, concat, lit
from pyspark.sql.types import IntegerType

# Create a SparkSession
spark = SparkSession.builder.appName("ETL").getOrCreate()

# Define the path to the input CSV file
input_path = "./data.csv"

# Read the input CSV file into a DataFrame
df = spark.read.format("csv").option("header", "true").load(input_path)

# Clean and transform the data
df_clean = df.withColumn("age", col("age").cast(IntegerType())) \
             .withColumn("gender", when(col("gender") == "M", "Male")
                                  .when(col("gender") == "F", "Female")
                                  .otherwise("Unknown")) \
             .withColumn("full_name", concat(col("first_name"), lit(" "), col("last_name"))) \
             .drop("first_name", "last_name")


# Print the schema and the DataFrame
df_clean.printSchema()
df_clean.show()

# Write the results to a Parquet file
output_path = "./data.parquet"
df_clean.write.mode("overwrite").format("parquet").save(output_path)

# Stop the SparkSession
spark.stop()