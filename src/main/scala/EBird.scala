/**
  * Created by abhinavmaurya on 4/19/17.
  */

import org.apache.spark.{SparkConf,SparkContext}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types.DoubleType;
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.types.{StructType, StructField, StringType, IntegerType}
import org.apache.spark.sql.functions._
import com.google.common.collect.ImmutableMap;
import org.apache.spark.sql.functions.countDistinct;

object EBird {

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("PageRank").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._
    // Load and parse the data file.
    val data = sc.textFile("/Users/vikasjanardhanan/courses/mreduce/project/test1.csv")

    //val parsedData = data.map { line =>
    //  val parts = line.split(',')
    //  parts.map(content =>
    //    if (content == "?")
    //      0.0
    //    else
    //      content)
    //}

    //print(parsedData);

    val df = sqlContext.read
      .format("com.databricks.spark.csv")
      .option("header", "true") // Use first line of all files as header
       // .option("nullValue","?")
      .option("inferSchema", "true") // Automatically infer data types
      .load("/Users/vikasjanardhanan/courses/mreduce/project/test.csv")

    df.show
    //val df1 = df.na.fill(0.0,df.columns)
    //df1.show
    val samplingIndexer = new StringIndexer()
      .setInputCol("SAMPLING_EVENT_ID")
      .setOutputCol("SAMPLING_EVENT_ID_INDEX")
      .fit(df)

    val indexed = samplingIndexer.transform(df).toDF()

    val LocIDIndexed = new StringIndexer()
      .setInputCol("LOC_ID")
      .setOutputCol("LOC_ID_INDEX")
      .fit(indexed)

    val dfLOCID = LocIDIndexed.transform(indexed).toDF()

    val BaileyIndexed = new StringIndexer()
      .setInputCol("BAILEY_ECOREGION")
      .setOutputCol("BAILEY_ECOREGION_INDEX")
      .fit(dfLOCID)

    val dfWithBailey = BaileyIndexed.transform(dfLOCID).toDF()
    val finalDF = dfWithBailey.drop("SAMPLING_EVENT_ID").drop("LOC_ID").drop("BAILEY_ECOREGION")

    val finalDFFill = finalDF.na.replace("*",ImmutableMap.of("?", "0.0"))
      .na.replace("Agelaius_phoeniceus",ImmutableMap.of("X", "1"))
    //val finalDFFill = finalDF.na.replace("Agelaius_phoeniceus",ImmutableMap.of("X", "1"))
    //val targedReplaced = finalDFFill.withColumn("Agelaius_phoeniceus",
    //  when(("bar" == "noValidFormat"), "noValue"))
    //val updatedDf = finalDF.withColumn("name", regexp_replace(col("name"), ",", ""))

    //val testDF = finalDFFill.select(finalDFFill("*").cast(DoubleType))
    val testDF = finalDFFill.columns.foldLeft(finalDFFill)((current, c) => current.withColumn(c, col(c).cast(DoubleType)))

    testDF.show
    print(testDF.schema)
    // create new dataframe with added column named "notempty"
    val finalDf = testDF.withColumn("Agelaius_phoeniceus",
      when($"Agelaius_phoeniceus" > 1.0, 1.0).otherwise(0.0))
    finalDf.show(100)
    finalDf.printSchema
    //print(finalDf.select(finalDf("Agelaius_phoeniceus")).distinct)
    finalDf.agg(countDistinct("Agelaius_phoeniceus")).toDF().show


    val ignored = List("Agelaius_phoeniceus")
    val featureIndex = finalDf.columns.diff(ignored).map(finalDF.columns.indexOf(_))
    val predictionIndex = finalDf.columns.indexOf("Agelaius_phoeniceus")

    val finalDFRDD = finalDf.rdd.map(r => LabeledPoint(
      r.getDouble(predictionIndex), // Get target value
    //  // Map feature indices to values
      Vectors.dense(featureIndex.map(r.getDouble(_)).toArray)
      //Vectors.dense(featureIndex.map(r.getDouble(_)).toArray)
    ))

    //print(finalDFRDD)
    val splits = finalDFRDD.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 10 // Use more in practice.
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    val impurity = "gini"
    val maxDepth = 9
    val maxBins = 4000

    val model = RandomForest.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
      numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

    //// Evaluate model on test instances and compute test error
    val labelAndPreds = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()
    println("Test Error = " + testErr)
    println("Learned classification forest model:\n" + model.toDebugString)

    // Save and load model
    model.save(sc, "target/tmp/myRandomForestClassificationModel")
    val sameModel = RandomForestModel.load(sc, "target/tmp/myRandomForestClassificationModel")
  }
  }
