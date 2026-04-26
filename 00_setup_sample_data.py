# Databricks notebook source
# MAGIC %md
# MAGIC # Sample Data Setup — Genie Code Practical Guide
# MAGIC
# MAGIC Run this notebook once to load the three sample datasets into Unity Catalog
# MAGIC so you can follow every example in the article.
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - Unity Catalog enabled workspace
# MAGIC - `CREATE TABLE` and `USE SCHEMA` privileges on your target catalog/schema
# MAGIC - The two CSV files uploaded to a Unity Catalog Volume (see Step 1)
# MAGIC
# MAGIC **Tables created:**
# MAGIC | Table | Rows | Description |
# MAGIC |-------|------|-------------|
# MAGIC | `prod.emea.sales_transactions` | 5,000 | EMEA sales with SKUs, timestamps, revenue, customer email (PII) |
# MAGIC | `prod.emea.customer_features` | 2,000 | Customer ML features + churn label |

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 0 — Configure your target catalog and schema
# MAGIC Edit the two variables below to match your environment.

# COMMAND ----------

TARGET_CATALOG = "prod"          # ← change if needed
TARGET_SCHEMA  = "emea"          # ← change if needed

# Fully qualified prefix used throughout
FQ = f"{TARGET_CATALOG}.{TARGET_SCHEMA}"

print(f"Target location: {FQ}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 1 — Create catalog and schema (if they don't exist)

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {TARGET_CATALOG}")
spark.sql(f"USE CATALOG {TARGET_CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {TARGET_SCHEMA}")
spark.sql(f"USE SCHEMA {TARGET_SCHEMA}")

print(f"Catalog '{TARGET_CATALOG}' and schema '{TARGET_SCHEMA}' are ready.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 2 — Upload the CSV files to a Unity Catalog Volume
# MAGIC
# MAGIC 1. In Catalog Explorer, navigate to **prod → emea**
# MAGIC 2. Click **Create → Volume** → name it `sample_data` → click **Create**
# MAGIC 3. Open the volume, click **Upload to this volume**
# MAGIC 4. Upload the two CSV files:
# MAGIC    - `sales_transactions.csv`
# MAGIC    - `customer_features.csv`
# MAGIC
# MAGIC Then update the path below:

# COMMAND ----------

VOLUME_PATH = f"/Volumes/{TARGET_CATALOG}/{TARGET_SCHEMA}/sample_data"

# Verify the files are there
import os
try:
    files = dbutils.fs.ls(VOLUME_PATH)
    print("Files found in volume:")
    for f in files:
        print(f"  {f.name}  ({f.size / 1024:.1f} KB)")
except Exception as e:
    print(f"ERROR: Could not list {VOLUME_PATH}")
    print(f"  → {e}")
    print("  Make sure you completed Step 2 above.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 3 — Load `sales_transactions`

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, TimestampType, DoubleType
)

sales_schema = StructType([
    StructField("transaction_id",  StringType(),    nullable=False),
    StructField("sku",             StringType(),    nullable=False),
    StructField("transaction_ts",  StringType(),    nullable=True),   # cast below
    StructField("revenue_eur",     DoubleType(),    nullable=True),
    StructField("store_id",        StringType(),    nullable=True),
    StructField("region",          StringType(),    nullable=False),
    StructField("customer_id",     StringType(),    nullable=True),
    StructField("customer_email",  StringType(),    nullable=True),   # PII — tagged below
])

sales_df = (
    spark.read
        .format("csv")
        .option("header", "true")
        .option("nullValue", "")
        .schema(sales_schema)
        .load(f"{VOLUME_PATH}/sales_transactions.csv")
        .withColumn(
            "transaction_ts",
            F.to_utc_timestamp(
                F.to_timestamp(F.col("transaction_ts"), "yyyy-MM-dd HH:mm:ss"),
                "Europe/Berlin"
            )
        )
)

print(f"Loaded {sales_df.count():,} rows from CSV")
print(f"Unique customers : {sales_df.select('customer_id').distinct().count():,}")
print(f"Unique emails    : {sales_df.select('customer_email').distinct().count():,}")
display(sales_df.limit(5))

# COMMAND ----------

# Write to Unity Catalog as a managed Delta table
(
    sales_df
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{FQ}.sales_transactions")
)

# ── Register table and column comments ────────────────────────────────────
spark.sql(f"""
    COMMENT ON TABLE {FQ}.sales_transactions IS
    'EMEA sales transactions sample dataset for Genie Code tutorial'
""")

spark.sql(f"ALTER TABLE {FQ}.sales_transactions ALTER COLUMN transaction_id COMMENT 'Unique transaction identifier (UUID)'")
spark.sql(f"ALTER TABLE {FQ}.sales_transactions ALTER COLUMN sku COMMENT 'Product stock-keeping unit'")
spark.sql(f"ALTER TABLE {FQ}.sales_transactions ALTER COLUMN transaction_ts COMMENT 'Timestamp of sale (UTC)'")
spark.sql(f"ALTER TABLE {FQ}.sales_transactions ALTER COLUMN revenue_eur COMMENT 'Transaction revenue in Euros — null or 0.0 for loyalty/internal orders'")
spark.sql(f"ALTER TABLE {FQ}.sales_transactions ALTER COLUMN store_id COMMENT 'Store reference key'")
spark.sql(f"ALTER TABLE {FQ}.sales_transactions ALTER COLUMN region COMMENT 'EMEA sub-region: DACH | BENELUX | NORDIC | IBERIA | EE'")
spark.sql(f"ALTER TABLE {FQ}.sales_transactions ALTER COLUMN customer_id COMMENT 'Customer reference ID — links to customer_features table'")
spark.sql(f"ALTER TABLE {FQ}.sales_transactions ALTER COLUMN customer_email COMMENT 'Customer email address — PII, must be masked before leaving bronze layer'")

# ── Tag customer_email as PII in Unity Catalog ────────────────────────────
# This tag is what the pii_columns_check.py helper reads to detect PII columns
# automatically — the PII masking skill in Genie Code depends on this being set
spark.sql(f"""
    ALTER TABLE {FQ}.sales_transactions
    ALTER COLUMN customer_email SET TAGS ('pii' = 'true')
""")

print(f"✓ {FQ}.sales_transactions written to Unity Catalog")
print(f"✓ customer_email tagged as pii=true in Unity Catalog")
spark.sql(f"SELECT COUNT(*) as row_count FROM {FQ}.sales_transactions").show()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 4 — Load `customer_features`

# COMMAND ----------

from pyspark.sql.types import IntegerType, DateType

features_schema = StructType([
    StructField("customer_id",             StringType(),  nullable=False),
    StructField("age",                     IntegerType(), nullable=True),
    StructField("tenure_months",           IntegerType(), nullable=True),
    StructField("num_products",            IntegerType(), nullable=True),
    StructField("avg_order_value_eur",     DoubleType(),  nullable=True),
    StructField("support_tickets_90d",     IntegerType(), nullable=True),
    StructField("days_since_last_order",   IntegerType(), nullable=True),
    StructField("region",                  StringType(),  nullable=True),
    StructField("event_date",              StringType(),  nullable=True),  # cast below
    StructField("churn_label",             IntegerType(), nullable=False),
])

features_df = (
    spark.read
        .format("csv")
        .option("header", "true")
        .schema(features_schema)
        .load(f"{VOLUME_PATH}/customer_features.csv")
        .withColumn("event_date", F.to_date(F.col("event_date"), "yyyy-MM-dd"))
)

print(f"Loaded {features_df.count():,} rows from CSV")
churn_rate = features_df.filter(F.col("churn_label") == 1).count() / features_df.count()
print(f"Churn rate: {churn_rate:.1%}")
display(features_df.limit(5))

# COMMAND ----------

(
    features_df
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("event_date")
    .saveAsTable(f"{FQ}.customer_features")
)

spark.sql(f"""
    COMMENT ON TABLE {FQ}.customer_features IS
    'Customer ML features with churn labels — partitioned by event_date'
""")

spark.sql(f"ALTER TABLE {FQ}.customer_features ALTER COLUMN customer_id COMMENT 'Unique customer identifier'")
spark.sql(f"ALTER TABLE {FQ}.customer_features ALTER COLUMN churn_label COMMENT 'Binary churn target: 1=churned within 90 days, 0=retained'")
spark.sql(f"ALTER TABLE {FQ}.customer_features ALTER COLUMN event_date COMMENT 'Feature snapshot date (partition key)'")
spark.sql(f"ALTER TABLE {FQ}.customer_features ALTER COLUMN avg_order_value_eur COMMENT 'Average basket value in Euros over customer lifetime'")
spark.sql(f"ALTER TABLE {FQ}.customer_features ALTER COLUMN support_tickets_90d COMMENT 'Number of support tickets raised in the last 90 days'")

print(f"✓ {FQ}.customer_features written to Unity Catalog")
spark.sql(f"SELECT COUNT(*) as row_count FROM {FQ}.customer_features").show()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 5 — Verify both tables in Unity Catalog

# COMMAND ----------

tables = spark.sql(f"SHOW TABLES IN {FQ}").filter(
    F.col("tableName").isin("sales_transactions", "customer_features")
)
display(tables)

# Quick row count summary
for tbl in ["sales_transactions", "customer_features"]:
    count = spark.sql(f"SELECT COUNT(*) as n FROM {FQ}.{tbl}").collect()[0]["n"]
    print(f"  {FQ}.{tbl:30s}  {count:,} rows")

# Confirm PII tag is set on customer_email
pii_check = spark.sql(f"""
    SELECT column_name, tag_name, tag_value
    FROM system.information_schema.column_tags
    WHERE catalog_name = '{TARGET_CATALOG}'
      AND schema_name  = '{TARGET_SCHEMA}'
      AND table_name   = 'sales_transactions'
      AND tag_name     = 'pii'
""")
print("\nPII-tagged columns in sales_transactions:")
pii_check.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ## ✅ Setup complete!
# MAGIC
# MAGIC Both tables are now registered in Unity Catalog under `prod.emea`.
# MAGIC You can verify them in **Catalog Explorer → prod → emea**.
# MAGIC
# MAGIC **Tables created:**
# MAGIC | Table | Rows | PII columns |
# MAGIC |-------|------|-------------|
# MAGIC | `prod.emea.sales_transactions` | 5,000 | `customer_email` (tagged `pii=true`) |
# MAGIC | `prod.emea.customer_features` | 2,000 | None |
# MAGIC
# MAGIC **Next step:** Open a new notebook and try your first Genie Code prompt:
# MAGIC ```
# MAGIC /findTables Find tables related to EMEA sales transactions
# MAGIC ```
# MAGIC
# MAGIC Genie Code will discover both tables and their column metadata automatically.
