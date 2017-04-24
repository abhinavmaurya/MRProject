/**
  * Created by abhinavmaurya on 4/19/17.
  */

import org.apache.spark.{SparkConf,SparkContext}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types.DoubleType;
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.sql.types.{StructType, StructField, StringType, IntegerType}
import org.apache.spark.sql.functions._
import com.google.common.collect.ImmutableMap;
import org.apache.spark.sql.functions.countDistinct;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}

/**
  * Creates model for eBird sighting prediction
  */
object BirdSightTrainer {


  def main(args: Array[String]) {

    /**
      * Loads all the columns in dataset required in the feature list.
      */
    val featuresInput = DatasetColumns.getColumnNameList.toBuffer - (DatasetColumns.getLabelColumnName, "SAMPLING_EVENT_ID")

    /**
      * Initial spark run configurations
      */
    val conf = new SparkConf().setAppName("EBird")
      //.setMaster("local[*]")
      .setMaster("yarn")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._


    /**
      * Load the raw dataframe
      */
    var rawDF = sqlContext.read
      .format("com.databricks.spark.csv")
      .option("header", "true") // Use first line of all files as header
      .option("NullValue","?")
      .option("inferSchema", "true") // Automatically infer data types
      //.load("/Users/vikasjanardhanan/courses/mreduce/project/test_1.csv")
      .load(args(0))

    /***
      * Select only those columns that are required to build the model
      */
    val colNos = DatasetColumns.getColumnIndexList.toBuffer - 0
    rawDF = rawDF.select(colNos map rawDF.columns map col: _*)

    /***
      * Replace all the null values in the dataframe with 0.0 and replace all X in label column to 1
      */
    var cleanedDF = rawDF.na.fill(0.0)
    cleanedDF = cleanedDF
      .na.replace("Agelaius_phoeniceus",ImmutableMap.of("X", "1"))

    /**
      * Cast all the columns to double that is suitable for input to randomforest.
      */
   var finalDF = cleanedDF.columns.foldLeft(cleanedDF)((current, c) => current.withColumn(c, col(c).cast(DoubleType)))

    /**
      * If label column value is >=1.0 replace with 1.0 so that the field has binary value indicating
      * if the bird was seen or not
      */
    finalDF = finalDF.withColumn("Agelaius_phoeniceus",
      when($"Agelaius_phoeniceus" >= 1.0, 1.0).otherwise(0.0))

    /**
      * Assembles the features in dataframe to a vector of features
      */
    val assembler = new VectorAssembler()
      .setInputCols(featuresInput.toArray)
      .setOutputCol("features")

    //val Array(trainingData, testData) = finalDf.withColumnRenamed("Agelaius_phoeniceus","label").randomSplit(Array(0.8, 0.2))

    /***
      * Create training data where label column is assigned to Agelaius_phoeniceus
      */
    val trainingData = finalDF.withColumnRenamed("Agelaius_phoeniceus","label")

    /**
      * Initialize randomforest object with num of trees and maxdepth as (31,20)
      */
     val rf = new RandomForestClassifier()
        .setLabelCol("label")
       .setFeaturesCol("features")
        .setNumTrees(31)
       .setMaxDepth(20)


    /**
      * Initialize pipeline to create randomforest model
      */
     val pipeline = new Pipeline()
      .setStages(Array(assembler,rf))

    /**
      * Create the model by fitting the training data
      */
     val model = pipeline.fit(trainingData)

    /*
     val predictions = model.transform(testData)

     val evaluator = new MulticlassClassificationEvaluator()
         .setLabelCol("label")
       .setPredictionCol("prediction")
       .setMetricName("accuracy")
     val accuracy = evaluator.evaluate(predictions)
     println("Test Error (31,20) with imputation all features= " + (1.0 - accuracy))
    */

   /* val paramGrid = new ParamGridBuilder()
      .addGrid(rf.numTrees,Array(25,28,32))
      //.addGrid(rf.numTrees,Array(125))
      //.addGrid(rf.maxDepth, Array(10))
      .addGrid(rf.maxDepth, Array(12,15,16))
      .addGrid(rf.impurity, Array("entropy", "gini"))
      .build()

     val cv = new CrossValidator()
       .setEstimator(pipeline)
       .setEvaluator(new MulticlassClassificationEvaluator)
       .setEstimatorParamMaps(paramGrid)
       .setNumFolds(3)  // Use 3+ in practice


     //val cvModel = trainValidationSplit.fit(trainingData)
     val cvModel = cv.fit(trainingData)
     val predictions_cv = cvModel.transform(testData)
     val accuracy_cv = evaluator.evaluate(predictions_cv)
     println("New Accuracy after selecting best model= " + accuracy_cv)*/


    /***
      * Save the model to input path
      */
    val rfModel = model.stages(1).asInstanceOf[RandomForestClassificationModel]
    //val rfModel = cvModel.write.overwrite().save(args(1))

    rfModel.write.overwrite().save(args(1))
  }
}
