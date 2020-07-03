package com.ysz.dm.mongodb;

public class QiniuTest_02 {

  public static void main(String[] args) throws Exception{
    PutPolicy putPolicy = new PutPolicy(bucketName);
    Mac mac = new Mac(accessKey, secretKey);
    String uptoken = putPolicy.token(mac);
    String localFile = "test.png";
    PutExtra extra = new PutExtra();
    PutRet ret = IoApi.putFile(uptoken, key, localFile, extra);

  }

}
