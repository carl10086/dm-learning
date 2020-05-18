package com.ysz.dm.mongodb;

import com.mongodb.client.MongoChangeStreamCursor;
import com.mongodb.client.MongoClient;
import com.mongodb.client.MongoClients;
import com.mongodb.client.MongoCollection;
import com.mongodb.client.MongoDatabase;
import com.mongodb.client.model.changestream.ChangeStreamDocument;
import com.mongodb.client.model.changestream.OperationType;
import org.bson.BsonDocument;
import org.bson.BsonString;
import org.bson.Document;

public class MongoStreamDm {

  public static void main(String[] args) throws Exception {
    MongoClient mongoClient = MongoClients.create("mongodb://10.1.5.188:27017");

    MongoDatabase test = mongoClient.getDatabase("meetings");
    MongoCollection<Document> streamTest = test.getCollection("stream_test");
    BsonDocument token = new BsonDocument();
    token.append("_data", new BsonString(
        "825EBCCAB6000000022B022C0100296E5A10049E189EB5CD5D45508FEE91DE2554CE8C46645F696400645EBCCAB626851C190202784D0004"));
    try (MongoChangeStreamCursor<ChangeStreamDocument<Document>> cursor = streamTest.watch()
        .resumeAfter(token)
        .cursor()) {
      while (cursor.hasNext()) {
        ChangeStreamDocument<Document> next = cursor.next();
        if (next.getOperationType() == OperationType.INSERT) {
          System.err.println("fullDocument:" + next.getFullDocument());
          System.err.println("documentKey:" + next.getDocumentKey());
          System.err.println("clusterTime:" + next.getClusterTime());
          System.err.println("token:" + next.getResumeToken());
        }
      }
    }

    mongoClient.close();
  }

}
