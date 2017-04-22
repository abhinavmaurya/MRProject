/**
  * Created by abhinavmaurya on 4/19/17.
  */

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types._
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.sql.functions._
import com.google.common.collect.ImmutableMap
import org.apache.spark.sql.functions.countDistinct
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

object BirdSightPredict {


  def main(args: Array[String]) {


    val featuresInput = DatasetColumns.getFeaturesColumnName.toBuffer - "SAMPLING_EVENT_ID"
    println(featuresInput)
    //val colNames = DatasetColumns.getColumnNameList.toArray

    val finalOuputColumns = Array("SAMPLING_EVENT_ID", "SAW_AGELAIUS_PHOENICEUS")

    val conf = new SparkConf().setAppName("EBird")
      .setMaster("local[*]")
      //.setMaster("yarn")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._
    // Load and parse the data file.
    var df = sqlContext.read
      .format("csv")
      .option("header", "true") // Use first line of all files as header
      .option("NullValue","?")
      .option("inferSchema", "true") // Automatically infer data types
      .load("subset_test.csv")
      //.load(args(0))

    /*val colNames = Array("SAMPLING_EVENT_ID","LATITUDE","LONGITUDE","MONTH","DAY","TIME","Agelaius_phoeniceus",
      "POP00_SQMI","HOUSING_DENSITY","HOUSING_PERCENT_VACANT","ELEV_GT"
      ,"BCR","OMERNIK_L3_ECOREGION","CAUS_TEMP_AVG","CAUS_TEMP_MIN","CAUS_TEMP_MAX",
      "CAUS_PREC","CAUS_SNOW")*/

    df.show()
    val colNos = DatasetColumns.getColumnIndexList
    df = df.select(colNos map df.columns map col: _*)
    df.count()

    df = df.toDF(DatasetColumns.getColumnNameList: _*)

    //df = df.select(colNames.head,colNames.tail: _*)

    df = df.withColumn("uniqueID",monotonically_increasing_id())

    df.show()

    // impute missing values
    val colsToBeImputed = Array("POP00_SQMI","HOUSING_DENSITY", "HOUSING_PERCENT_VACANT", "CAUS_TEMP_AVG", "CAUS_TEMP_MIN", "CAUS_TEMP_MAX", "CAUS_PREC","CAUS_SNOW","OMERNIK_L3_ECOREGION","BCR","ELEV_GT")
    val categoricalCol = Array("ELEV_GT","CAUS_TEMP_AVG", "CAUS_TEMP_MIN", "CAUS_TEMP_MAX", "CAUS_PREC","CAUS_SNOW")
    val imputedVal = df.select(colsToBeImputed.map(avg(_)):_*).toDF(colsToBeImputed:_*)

    val imputerMap = imputedVal.columns.zip(imputedVal.first().toSeq).
      map(a =>
        if (categoricalCol contains a._1)
          (a._1 -> a._2.toString.toDouble.ceil.toString)
        else
          (a._1 -> a._2)).toMap

    val imputedDF = df.na.fill(imputerMap)
    imputedDF.show



    val finalDFFill = imputedDF//.na.replace("Agelaius_phoeniceus",ImmutableMap.of("X", "1"))

    val dfColExceptSampling = finalDFFill.columns.toBuffer - "SAMPLING_EVENT_ID"
    println(dfColExceptSampling)
    val testDF = dfColExceptSampling.foldLeft(finalDFFill)((current, c) => current.withColumn(c, col(c).cast(DoubleType)))

    // create new dataframe with added column named "notempty"
    val finalDf = testDF/*.withColumn("Agelaius_phoeniceus",
      when($"Agelaius_phoeniceus" > 1.0, 1.0).otherwise(0.0))*/
    //finalDf.show(100)
    //finalDf.printSchema
    val assembler = new VectorAssembler()
      .setInputCols(featuresInput.toArray)
      .setOutputCol("features")

    val testData = assembler.transform(finalDf)

    // Retrieve the RandomForest model from hdfs
    val rfModelFile = RandomForestClassificationModel.load("model")

    // Chain indexers and forest in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(assembler, rfModelFile))

    // Makes prediction and order it by index.
    val predictions = rfModelFile.transform(testData).orderBy("uniqueID")

    predictions.show()


    //Selects the required columns
    val finalOutput = predictions.select(col("SAMPLING_EVENT_ID"),col("prediction").cast(DataTypes.IntegerType)).toDF(finalOuputColumns: _*)

    finalOutput.show(100)
    //Creates the output in the required file format
    finalOutput.coalesce(1).write.format("com.databricks.spark.csv").option("header","true").save("output")
  }
}
