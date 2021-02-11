


import org.apache.hadoop.hbase._
import org.apache.hadoop.hbase.mapreduce.TableInputFormat
import org.apache.hadoop.hbase.util.Bytes
import org.apache.spark.sql.{SQLContext, SaveMode}
import org.apache.spark.{SparkConf, SparkContext}

object DataProcess {

  //需要传递的参数为 保存文件类型，id，属性，属性最小值，属性最大值
  def main(args: Array[String]) {


    var fileType = args(0)
    var id = args(1)
    var property = args(2)
    var min = args(3)
    var max = args(4)
    var startTime = args(5)
    var stopTime = args(6)




    val hbaseConf = HBaseConfiguration.create()
    val sc = new SparkContext(new SparkConf()
      .setAppName("SparkWriteHBase")
      .setMaster("spark://node1:7077")
      //  .setMaster("local[*]")
      //.setJars(List("/Users/denghanbo/IdeaProjects/MLlib2/out/artifacts/dataproduct/MLlib2.jar"))
    )
    //设置查询的表名
    hbaseConf.set(TableInputFormat.INPUT_TABLE, "transformer")

    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._



    val hbaseRDD = sc.newAPIHadoopRDD(hbaseConf, classOf[TableInputFormat],
      classOf[org.apache.hadoop.hbase.io.ImmutableBytesWritable],
      classOf[org.apache.hadoop.hbase.client.Result])




    //遍历输出
//    hbaseRDD.foreach({ case (_,result) =>
//      val key = Bytes.toString(result.getRow)
//      val id = Bytes.toString(result.getValue("D".getBytes,"id".getBytes))
//      val f = Bytes.toString(result.getValue("D".getBytes,"f".getBytes))
//      val t = Bytes.toString(result.getValue("D".getBytes,"t".getBytes))
//      println("Row key:"+key+" id:"+id+" f:"+f+" t:"+t)
//    })





    // 将数据映射为表  也就是将 RDD转化为 dataframe schema
    val dataRDD = hbaseRDD.map(r=>(
      Bytes.toString(r._2.getRow()),

      Bytes.toString(r._2.getValue(Bytes.toBytes("D"),Bytes.toBytes("environmentT"))),
      Bytes.toString(r._2.getValue(Bytes.toBytes("D"),Bytes.toBytes("oilT"))),
      Bytes.toString(r._2.getValue(Bytes.toBytes("D"),Bytes.toBytes("windingStrain"))),
      Bytes.toString(r._2.getValue(Bytes.toBytes("D"),Bytes.toBytes("noise"))),
      Bytes.toString(r._2.getValue(Bytes.toBytes("D"),Bytes.toBytes("windingFault"))),
      Bytes.toString(r._2.getValue(Bytes.toBytes("D"),Bytes.toBytes("coreFault"))),
      Bytes.toString(r._2.getValue(Bytes.toBytes("D"),Bytes.toBytes("engery0"))),
      Bytes.toString(r._2.getValue(Bytes.toBytes("D"),Bytes.toBytes("engery1"))),
      Bytes.toString(r._2.getValue(Bytes.toBytes("D"),Bytes.toBytes("engery2"))),
      Bytes.toString(r._2.getValue(Bytes.toBytes("D"),Bytes.toBytes("engery3"))),
      Bytes.toString(r._2.getValue(Bytes.toBytes("D"),Bytes.toBytes("engery4"))),
      Bytes.toString(r._2.getValue(Bytes.toBytes("D"),Bytes.toBytes("engery5"))),
      Bytes.toString(r._2.getValue(Bytes.toBytes("D"),Bytes.toBytes("engery6"))),
      Bytes.toString(r._2.getValue(Bytes.toBytes("D"),Bytes.toBytes("engery7"))),
      Bytes.toString(r._2.getValue(Bytes.toBytes("D"),Bytes.toBytes("fData")))
    )).toDF("id_date","environmentT","oilT","windingStrain","noise","windingFault","coreFault","engery0","engery1","engery2","engery3","engery4","engery5","engery6","engery7","fData")

    dataRDD.registerTempTable("DATA")


    //导出指定属性的数据
    //val df2 = sqlContext.sql("SELECT substring(id_date,4,14),"+property+" FROM DATA WHERE  id_date > "+id+startTime+" AND id_date < "+id+stopTime+ " AND "+property+">"+min+" AND "+property+"<"+max)

    //导出所有数据
    val df2 = sqlContext.sql("SELECT * FROM DATA WHERE  id_date > "+id+startTime+" AND id_date < "+id+stopTime)
    df2.show(false)

    df2.write.format(fileType).mode(SaveMode.Overwrite).save("hdfs://node1:9000/transformer/data/")


  }




}
