package classer

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SparkSession}


object RandomForest {


  def PreProcess2(path:String):DataFrame={

    val sparkSession = SparkSession.builder().master("local[*]").appName("LogisticRegression").getOrCreate()



    //读到的数据有缺失
    val data = sparkSession.read.format("csv")
      .option("delimiter"," ")
      .option("inferSchema", "true")
      //"/Users/denghanbo/Documents/mldata/adultTest/adult.data.txt"
      .load(path)
      .toDF("f1","f2","f3","f4","f5","f6","label")



    //特征合并
    val assembler = new VectorAssembler()
      .setInputCols(Array("f2","f3","f4","f5","f6"))

      .setOutputCol("features")


    val data2 = assembler.transform(data)



    val indexer1 = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("labelIndex")

    val data3 = indexer1.fit(data2).transform(data2)


    val encoder = new OneHotEncoder()
      .setInputCol("labelIndex")
      .setOutputCol("labelIndexVec")
    val data4 = encoder.transform(data3)


    data4.show(false)



    return data3






  }

  def PreProcess(param1: String): DataFrame = {


    val sparkSession = SparkSession.builder().master("local[*]").appName("RandomForest").getOrCreate()

    val data = sparkSession.read.format("csv")
      .option("delimiter",",")
      .option("inferSchema", "true")
      .load("/Users/denghanbo/Documents/mldata/adultTest/adult.data.txt")
      .toDF("f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","f10","f11","f12","f13","f14")



    //把离散特征数字化
    val s2i1 = new StringIndexer()
      .setInputCol("f1")
      .setOutputCol("f1Index")

    val encoder1 = new OneHotEncoder()
      .setInputCol("f1Index")
      .setOutputCol("f1Vec")




    val s2i3 = new StringIndexer()
      .setInputCol("f3")
      .setOutputCol("f3Index")

    val encoder3 = new OneHotEncoder()
      .setInputCol("f3Index")
      .setOutputCol("f3Vec")





    val s2i5 = new StringIndexer()
      .setInputCol("f5")
      .setOutputCol("f5Index")

    val encoder5 = new OneHotEncoder()
      .setInputCol("f5Index")
      .setOutputCol("f5Vec")




    val s2i6 = new StringIndexer()
      .setInputCol("f6")
      .setOutputCol("f6Index")


    val encoder6 = new OneHotEncoder()
      .setInputCol("f6Index")
      .setOutputCol("f6Vec")





    val s2i7 = new StringIndexer()
      .setInputCol("f7")
      .setOutputCol("f7Index")

    val encoder7 = new OneHotEncoder()
      .setInputCol("f7Index")
      .setOutputCol("f7Vec")






    val s2i8 = new StringIndexer()
      .setInputCol("f8")
      .setOutputCol("f8Index")

    val encoder8 = new OneHotEncoder()
      .setInputCol("f8Index")
      .setOutputCol("f8Vec")




    val s2i9 = new StringIndexer()
      .setInputCol("f9")
      .setOutputCol("f9Index")

    val encoder9 = new OneHotEncoder()
      .setInputCol("f9Index")
      .setOutputCol("f9Vec")




    val s2i13 = new StringIndexer()
      .setInputCol("f13")
      .setOutputCol("f13Index")

    val encoder13 = new OneHotEncoder()
      .setInputCol("f13Index")
      .setOutputCol("f13Vec")





    val s2i14 = new StringIndexer()
      .setInputCol("f14")
      .setOutputCol("label")





    val pipeline = new Pipeline()
      .setStages(Array(s2i1,encoder1,s2i3,encoder3,s2i5,encoder5,s2i6,encoder6,s2i7,encoder7,s2i8,encoder8,s2i9,encoder9,s2i13,encoder13,s2i14))

    val data1 = pipeline.fit(data).transform(data)




    //特征合并
    val assembler = new VectorAssembler()
      .setInputCols(Array("f0","f1Vec","f2","f3Vec","f4","f5Vec","f6Vec","f7Vec","f8Vec","f9Vec","f10","f11","f12","f13Vec"))
      //.setInputCols(Array("f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","f10","f11","f12","f13"))
      .setOutputCol("features")

    val data2 = assembler.transform(data1)



    return data2
  }

  def PreProccess1(path: String): DataFrame = {


    val sparkSession = SparkSession.builder().master("spark://192.168.0.108:7077").appName("RandomForestClassifier").getOrCreate()


    //读到的数据有缺失
    val data = sparkSession.read.format("csv")
      .option("delimiter",",")
      .option("inferSchema", "true")
      //"/Users/denghanbo/Documents/mldata/adultTest/adult.data.txt"
      .load(path)
      .toDF("f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","f10","f11","f12","f13","f14")






    //进行差值，对于离散变量就随便差一个值，连续变量差均值

    val newData = data
      //      .filter(data("f0")=!=" ?")
      .filter(data("f1")=!=" ?")
      //      .filter(data("f2")=!=" ?")
      .filter(data("f3")=!=" ?")
      //      .filter(data("f4")=!=" ?")
      .filter(data("f5")=!=" ?")
      .filter(data("f6")=!=" ?")
      .filter(data("f7")=!=" ?")
      .filter(data("f8")=!=" ?")
      .filter(data("f9")=!=" ?")
      //      .filter(data("f10")=!=" ?")
      //      .filter(data("f11")=!=" ?")
      //      .filter(data("f12")=!=" ?")
      .filter(data("f13")=!=" ?")




    //  newData.write.format("csv").save("/Users/denghanbo/Documents/out.csv")




    //把离散特征数字化
    val s2i1 = new StringIndexer()
      .setInputCol("f1")
      .setOutputCol("f1Index")

    val encoder1 = new OneHotEncoder()
      .setInputCol("f1Index")
      .setOutputCol("f1Vec")




    val s2i3 = new StringIndexer()
      .setInputCol("f3")
      .setOutputCol("f3Index")

    val encoder3 = new OneHotEncoder()
      .setInputCol("f3Index")
      .setOutputCol("f3Vec")





    val s2i5 = new StringIndexer()
      .setInputCol("f5")
      .setOutputCol("f5Index")

    val encoder5 = new OneHotEncoder()
      .setInputCol("f5Index")
      .setOutputCol("f5Vec")




    val s2i6 = new StringIndexer()
      .setInputCol("f6")
      .setOutputCol("f6Index")


    val encoder6 = new OneHotEncoder()
      .setInputCol("f6Index")
      .setOutputCol("f6Vec")





    val s2i7 = new StringIndexer()
      .setInputCol("f7")
      .setOutputCol("f7Index")

    val encoder7 = new OneHotEncoder()
      .setInputCol("f7Index")
      .setOutputCol("f7Vec")






    val s2i8 = new StringIndexer()
      .setInputCol("f8")
      .setOutputCol("f8Index")

    val encoder8 = new OneHotEncoder()
      .setInputCol("f8Index")
      .setOutputCol("f8Vec")




    val s2i9 = new StringIndexer()
      .setInputCol("f9")
      .setOutputCol("f9Index")

    val encoder9 = new OneHotEncoder()
      .setInputCol("f9Index")
      .setOutputCol("f9Vec")




    val s2i13 = new StringIndexer()
      .setInputCol("f13")
      .setOutputCol("f13Index")

    val encoder13 = new OneHotEncoder()
      .setInputCol("f13Index")
      .setOutputCol("f13Vec")





    val s2i14 = new StringIndexer()
      .setInputCol("f14")
      .setOutputCol("label")







    //
    //    //尺度规范化
    //    val scaler2 = new MaxAbsScaler()
    //      .setInputCol("f2")
    //      .setOutputCol("f2scaled")
    //
    //
    //
    //
    //    val scaler4 = new MinMaxScaler()
    //      .setInputCol("f4")
    //      .setOutputCol("f4scaled")
    //      .setMax(100)
    //      .setMin(0)
    //
    //    val scaler10 = new MinMaxScaler()
    //      .setInputCol("f10")
    //      .setOutputCol("f10scaled")
    //      .setMax(100)
    //      .setMin(0)
    //
    //    val scaler11 = new MinMaxScaler()
    //      .setInputCol("f11")
    //      .setOutputCol("f11scaled")
    //      .setMax(100)
    //      .setMin(0)
    //
    //    val scaler12 = new MinMaxScaler()
    //      .setInputCol("f12")
    //      .setOutputCol("f12scaled")
    //      .setMax(100)
    //      .setMin(0)



    val pipeline = new Pipeline()
      //.setStages(Array(s2i1,encoder1,s2i3,encoder3,s2i5,encoder5,s2i6,encoder6,s2i7,encoder7,s2i8,encoder8,s2i9,encoder9,s2i13,encoder13,s2i14,scaler2,scaler4,scaler10,scaler11,scaler12))
      .setStages(Array(s2i1,encoder1,s2i3,encoder3,s2i5,encoder5,s2i6,encoder6,s2i7,encoder7,s2i8,encoder8,s2i9,encoder9,s2i13,encoder13,s2i14))

    val data1 = pipeline.fit(newData).transform(newData)




    //特征合并
    val assembler = new VectorAssembler()
      .setInputCols(Array("f0","f1Vec","f2","f3Vec","f4","f5Vec","f6Vec","f7Vec","f8Vec","f9Vec","f10","f11","f12","f13Vec"))
      //.setInputCols(Array("f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","f10","f11","f12","f13"))
      .setOutputCol("features")


    val data2 = assembler.transform(data1)


    //data2.show(false)



    return data2
  }

  def main(args: Array[String]): Unit = {


    //val  data2 = PreProccess1("hdfs://192.168.0.108:9000/h1/adult.data.txt")
    val data2 = PreProcess2("/Users/denghanbo/Desktop/DATA.txt")


    val Array(trainingData, testData) = data2.randomSplit(Array(0.7, 0.3))

    // Train a RandomForest model.
    val rf = new RandomForestClassifier()
      .setLabelCol("labelIndex")
      .setFeaturesCol("features")
      .setNumTrees(500)



    // Make predictions.
    val predictions = rf.fit(trainingData).transform(testData)

    // Select example rows to display.
    predictions.show(false)

    // Select (prediction, true label) and compute test error.
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("labelIndex")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println("Test Correct rate = " +accuracy)

  }


}
