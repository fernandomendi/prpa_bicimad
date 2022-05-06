from math import acos, cos, radians, sin
import random
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def get_min_max_lat_lon(df):
    min_lat = df.agg({"latitude":"min"}).head()["min(latitude)"]
    max_lat = df.agg({"latitude":"max"}).head()["max(latitude)"]
    min_lon = df.agg({"longitude":"min"}).head()["min(longitude)"]
    max_lon = df.agg({"longitude":"max"}).head()["max(longitude)"]
    return min_lat, max_lat, min_lon, max_lon

def generate_points(n, min_lat, max_lat, min_lon, max_lon):
    pts = []
    for _ in range(1000):
        lat = random.uniform(min_lat, max_lat)
        lon = random.uniform(min_lon, max_lon)
        pts.append((lat, lon))
    return sc.parallelize(pts)

# def is_close_to_stop(pt):


def get_coords_from_sid(sid, df):
    row = df.filter(df.id == sid).head()
    return (row["latitude"], row["longitude"])

def distance_sphere(pt1, pt2):
    pt1, pt2 = map(radians, [pt1, pt2])
    return 6371 * (acos(sin(pt1[0]) * sin(pt2[0]) + cos(pt1[0]) * cos(pt2[0]) * cos(pt1[1] - pt2[1])))

if __name__ == "__main__":
    spark = SparkSession.builder.getOrCreate()
    sc = spark.sparkContext

    df = spark.read.json("202106.json")
    stops_df = spark.createDataFrame(df.first()["stations"])
    # stops_rdd = stops_df.rdd.map(lambda x: (x["id"], x["latitude"], x["longitude"]))
    reduced_stops_df = stops_df.select(col("id"), col("latitude").cast("float"), col("longitude").cast("float"))

