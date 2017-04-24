/**
  * Created by abhinavmaurya on 4/19/17.
  */

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.types._
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler

object BirdSightPredict {


  def main(args: Array[String]) {

    /**
      * Loads all the columns in dataset required in the feature list.
      */
    val featuresInput = DatasetColumns.getColumnNameList.toBuffer - (DatasetColumns.getLabelColumnName, "SAMPLING_EVENT_ID")

    /**
      * Final column list in the predicted output
      */
    val finalOuputColumns = Array("SAMPLING_EVENT_ID", "SAW_AGELAIUS_PHOENICEUS")

    /**
      * Initial spark run configurations
      */
    val conf = new SparkConf().setAppName("EBird")
      //.setMaster("local[*]")
      .setMaster("yarn")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    /**
      * Load the raw dataframe
      */
    var df = sqlContext.read
      .format("csv")
      .option("header", "true") // Use first line of all files as header
      .option("NullValue","?")
      .option("inferSchema", "true") // Automatically infer data types
      //.load("subset_test.csv")
      .load(args(0))

    /***
      * Select only those columns that are required to predict the sighting
      */
    val colNos = DatasetColumns.getColumnIndexList
    df = df.select(colNos map df.columns map col: _*)
    df = df.toDF(DatasetColumns.getColumnNameList: _*)

    /***
      * Add monotonically increasing sequence id to dataframe to get the final output in sorted order.
      */
    df = df.withColumn("uniqueID",monotonically_increasing_id())
    /***
      * Replace all the null values in the dataframe with 0.0
      */
    val cleanedDF = df.na.fill(0.0)


    val dfColExceptSampling = cleanedDF.columns.toBuffer - "SAMPLING_EVENT_ID"
    /**
      * Cast all the columns to double that is suitable for input to randomforest.
      */
    val finalDf = dfColExceptSampling.foldLeft(cleanedDF)((current, c) => current.withColumn(c, col(c).cast(DoubleType)))
    finalDf.show
    /**
      * Assembles the features in dataframe to a vector of features
      */
    val assembler = new VectorAssembler()
      .setInputCols(featuresInput.toArray)
      .setOutputCol("features")

    val testData = assembler.transform(finalDf)

    /**
      *Retrieve the RandomForest model from hdfs
       */
    val rfModelFile = RandomForestClassificationModel.load(args(1))

    /**Chain indexers and forest in a Pipeline.
      *
      */
    val pipeline = new Pipeline()
      .setStages(Array(assembler, rfModelFile))

    /**
      * Makes prediction and order it by index.
       */
    val predictions = rfModelFile.transform(testData).orderBy("uniqueID")


    //Selects the required columns
    val finalOutput = predictions.select(col("SAMPLING_EVENT_ID"),col("prediction").cast(DataTypes.IntegerType)).toDF(finalOuputColumns: _*)

    finalOutput.show(100)
    //Creates the output in the required file format
    //finalOutput.coalesce(1).write.format("com.databricks.spark.csv").option("header","true").save("output")
    finalOutput.coalesce(1).write.format("com.databricks.spark.csv").option("header","true").save(args(2))
  }
}
