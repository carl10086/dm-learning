package com.ysz.dm.fast.kafka.adm;

import java.util.Map;
import java.util.Map.Entry;
import java.util.Properties;
import org.apache.kafka.clients.admin.AdminClient;
import org.apache.kafka.clients.admin.KafkaAdminClient;
import org.apache.kafka.clients.admin.ListConsumerGroupOffsetsResult;
import org.apache.kafka.clients.consumer.OffsetAndMetadata;
import org.apache.kafka.common.TopicPartition;

public class KafkaAdm {

  public static void main(String[] args) throws Exception {
    Properties props = new Properties();
    props
        .put("bootstrap.servers", "10.1.5.129:9092,10.1.5.166:9092,10.1.5.182:9092"); // Kafka集群在那裡?
    AdminClient adminClient = KafkaAdminClient.create(props);
    ListConsumerGroupOffsetsResult offsetsResult = adminClient.listConsumerGroupOffsets(
        "dt_audit_worker_v1"
    );
    Map<TopicPartition, OffsetAndMetadata> metadataMap = offsetsResult
        .partitionsToOffsetAndMetadata().get();

    for (Entry<TopicPartition, OffsetAndMetadata> entry : metadataMap
        .entrySet()) {
      System.err.println(entry);
    }

  }


}
