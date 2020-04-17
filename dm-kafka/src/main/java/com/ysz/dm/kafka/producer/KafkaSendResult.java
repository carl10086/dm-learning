package com.ysz.dm.kafka.producer;

import org.apache.kafka.clients.producer.RecordMetadata;

public class KafkaSendResult {

  private final RecordMetadata recordMetadata;

  public KafkaSendResult(RecordMetadata recordMetadata) {
    this.recordMetadata = recordMetadata;
  }

  @Override
  public String toString() {
    return "KafkaSendResult{" +
        "topic=" + recordMetadata.topic() + ", " +
        "offset=" + recordMetadata.offset() + ", " +
        "partition=" + recordMetadata.partition() + ", " +
        '}';
  }
}
