package classer

import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

object Bayes {

  def main(args: Array[String]): Unit = {


    val conf = new SparkConf().setAppName("WordCount").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val sparkSession =  SparkSession.builder().getOrCreate()

    // Load the data stored in LIBSVM format as a DataFrame.
    val data = sparkSession.read.format("csv")
      .option("delimiter", ",")
      .option("inferSchema",true)
      .load("/Users/denghanbo/Documents/mldata/inputdata3.txt")
      .toDF("f1","f2","f3","f4","label")


    //特征f1-f4合并
    val assembler = new VectorAssembler()
      .setInputCols(Array("f1", "f2", "f3", "f4"))
      .setOutputCol("features")

    val data1 = assembler.transform(data)





    val s2i = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("labelNumber")
      .fit(data1)

    val data2 = s2i.transform(data1)




    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = data2.randomSplit(Array(0.7, 0.3))


    val model = new NaiveBayes()
      .setLabelCol("labelNumber")
      .setSmoothing(10)
      .fit(trainingData)




    val predictions = model.transform(testData)




    val i2s = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictionLabel")
      .setLabels(s2i.labels)



    val  newPredictions = i2s.transform(predictions)

    newPredictions.show(false)

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("labelNumber")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(newPredictions)
    println("Accuracy: " + accuracy)

  }



}
