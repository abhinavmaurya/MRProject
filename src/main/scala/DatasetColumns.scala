/**
  * Created by abhinavmaurya on 4/21/17.
  */

object DatasetColumns {
  val columnNameIndexPairs = List(
    ("SAMPLING_EVENT_ID", 0),
    ("LATITUDE",2),
    ("LONGITUDE",3),
    ("MONTH",5),
    ("DAY",6),
    ("TIME",7),
    //("Agelaius_phoeniceus",26),
    ("POP00_SQMI",955),
    ("HOUSING_DENSITY",956),
    ("HOUSING_PERCENT_VACANT",957),
    ("ELEV_GT",958),
    ("BCR",960),
    ("OMERNIK_L3_ECOREGION",962),
    ("CAUS_TEMP_AVG",963),
    ("CAUS_TEMP_MIN",964),
    ("CAUS_TEMP_MAX",965),
    ("CAUS_PREC",966),
    ("CAUS_SNOW",967)
  )

  /*def main(args: Array[String]){

    //val newList = columns.map(tuple=>tuple._1)
    print(getColumnIndexList())
  }*/

  def getColumnIndexList = columnNameIndexPairs.map(tuple=>tuple._2)

  def getColumnNameList= columnNameIndexPairs.map(tuple=>tuple._1)

  def getColumnNameIndexPairList = columnNameIndexPairs

  def getLabelColumnName = "Agelaius_phoeniceus"

  def getLabelColumnIndex = 26

  def getFeaturesColumnName = columnNameIndexPairs.map(tuple=>tuple._1).filter(name => name!=getLabelColumnName)

  def getFeaturesColumnIndex = columnNameIndexPairs.map(tuple=>tuple._2).filter(name => name!=getLabelColumnIndex)

}
