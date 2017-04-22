/**
  * Created by abhinavmaurya on 4/21/17.
  */

object DatasetColumns {
  val columnNameIndexPairs = List(
    //("SAMPLING_EVENT_ID", 0),
    ("LATITUDE",2),
    ("LONGITUDE",3),
    ("MONTH",5),
    ("DAY",6),
    ("TIME",7),
    ("Agelaius_phoeniceus",26),
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
    ("CAUS_SNOW",967),
    ("NLCD2001_FS_C11_7500_PLAND",968),
    //("NLCD2001_FS_C12_7500_PLAND",969),
    ("NLCD2001_FS_C21_7500_PLAND",970),
    ("NLCD2001_FS_C22_7500_PLAND",971),
    ("NLCD2001_FS_C23_7500_PLAND",972),
    ("NLCD2001_FS_C24_7500_PLAND",973),
    ("NLCD2001_FS_C31_7500_PLAND",974),
    ("NLCD2001_FS_C41_7500_PLAND",975),
    ("NLCD2001_FS_C42_7500_PLAND",976),
    ("NLCD2001_FS_C43_7500_PLAND",977),
    ("NLCD2001_FS_C52_7500_PLAND",978),
    ("NLCD2001_FS_C71_7500_PLAND",979),
    ("NLCD2001_FS_C81_7500_PLAND",980),
    ("NLCD2001_FS_C82_7500_PLAND",981),
    ("NLCD2001_FS_C90_7500_PLAND",982),
    ("NLCD2001_FS_C95_7500_PLAND",983),
    ("NLCD2006_FS_C11_7500_PLAND",984),
    //("NLCD2006_FS_C12_7500_PLAND",985),
    ("NLCD2006_FS_C21_7500_PLAND",986),
    ("NLCD2006_FS_C22_7500_PLAND",987),
    ("NLCD2006_FS_C23_7500_PLAND",988),
    ("NLCD2006_FS_C24_7500_PLAND",989),
    ("NLCD2006_FS_C31_7500_PLAND",990),
    ("NLCD2006_FS_C41_7500_PLAND",991),
    ("NLCD2006_FS_C42_7500_PLAND",992),
    ("NLCD2006_FS_C43_7500_PLAND",993),
    ("NLCD2006_FS_C52_7500_PLAND",994),
    ("NLCD2006_FS_C71_7500_PLAND",995),
    ("NLCD2006_FS_C81_7500_PLAND",996),
    ("NLCD2006_FS_C82_7500_PLAND",997),
    ("NLCD2006_FS_C90_7500_PLAND",998),
    ("NLCD2006_FS_C95_7500_PLAND",999),
    ("NLCD2011_FS_C11_7500_PLAND",1000),
    ("NLCD2011_FS_C12_7500_PLAND",1001),
    ("NLCD2011_FS_C21_7500_PLAND",1002),
    ("NLCD2011_FS_C22_7500_PLAND",1003),
    ("NLCD2011_FS_C23_7500_PLAND",1004),
    ("NLCD2011_FS_C24_7500_PLAND",1005),
    ("NLCD2011_FS_C31_7500_PLAND",1006),
    ("NLCD2011_FS_C41_7500_PLAND",1007),
    ("NLCD2011_FS_C42_7500_PLAND",1008),
    ("NLCD2011_FS_C43_7500_PLAND",1009),
    ("NLCD2011_FS_C52_7500_PLAND",1010),
    ("NLCD2011_FS_C71_7500_PLAND",1011),
    ("NLCD2011_FS_C81_7500_PLAND",1012),
    ("NLCD2011_FS_C82_7500_PLAND",1013),
    ("NLCD2011_FS_C90_7500_PLAND",1014),
    ("NLCD2011_FS_C95_7500_PLAND",1015)
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
