package com.ysz.dm.mongodb;

import com.mongodb.client.FindIterable;
import com.mongodb.client.MongoClient;
import com.mongodb.client.MongoClients;
import com.mongodb.client.MongoDatabase;
import java.util.Date;
import java.util.Objects;
import org.apache.commons.lang3.time.DateUtils;
import org.bson.Document;

/**
 * @author carl
 */
public class Mongo_op_log_Dm {


  public static void main(String[] args) throws Exception {
    new Mongo_op_log_Dm().findOplog();
  }

  private void findOplog() {
    System.err.println(DateUtils.addHours(new Date(), -20).getTime() / 1000L);

    MongoClient mongoClient = MongoClients.create("mongodb://10.1.5.71:27017");
    MongoDatabase local = mongoClient.getDatabase("local");
    FindIterable<Document> documents = local.getCollection("oplog.rs").find();
    for (Document document : documents) {
      final String ns = document.getString("ns");
      if (ns != null && Objects.equals(ns, "copyright.copyright_user")) {
        System.err.println(document);
      }
    }
  }

}
