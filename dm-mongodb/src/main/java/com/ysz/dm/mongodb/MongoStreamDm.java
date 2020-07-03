package com.ysz.dm.mongodb;

import com.mongodb.client.MongoClient;
import com.mongodb.client.MongoClients;

public class MongoStreamDm {

  public static void main(String[] args) throws Exception {
    MongoClient mongoClient = MongoClients.create("mongodb://10.1.5.188:27017");

    Thread.sleep(1000L);
    mongoClient.close();
  }

}
