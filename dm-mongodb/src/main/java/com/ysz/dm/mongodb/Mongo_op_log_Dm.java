package com.ysz.dm.mongodb;

import com.mongodb.client.FindIterable;
import com.mongodb.client.MongoClient;
import com.mongodb.client.MongoClients;
import com.mongodb.client.MongoDatabase;
import java.util.Date;
import org.apache.commons.lang3.time.DateUtils;
import org.bson.Document;
import org.springframework.data.mongodb.core.MongoTemplate;

/**
 * @author carl
 */
public class Mongo_op_log_Dm {


  public static void main(String[] args) throws Exception {
    new Mongo_op_log_Dm().findOplog();
  }

  private void findOplog() {
    System.err.println(DateUtils.addHours(new Date(), -20).getTime() / 1000L);

    MongoClient mongoClient = MongoClients.create("mongodb://10.1.5.196:27018");
    MongoTemplate mongoTemplate = new MongoTemplate(
        mongoClient
        , "local");

    MongoDatabase local = mongoClient.getDatabase("local");
    FindIterable<Document> documents = local.getCollection("oplog.rs").find();
    for (Document document : documents) {
      System.out.println(document);
    }
  }

}
