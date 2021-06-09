package com.ysz.dm.fast.kafka.consumer;

import com.ysz.dm.fast.kafka.KafkaProducerDm;
import java.time.Duration;
import java.util.Collection;
import java.util.Collections;
import java.util.Properties;
import org.apache.kafka.clients.consumer.ConsumerRebalanceListener;
import org.apache.kafka.clients.consumer.ConsumerRecord;
import org.apache.kafka.clients.consumer.ConsumerRecords;
import org.apache.kafka.clients.consumer.KafkaConsumer;
import org.apache.kafka.common.TopicPartition;

public class KafkaTsgConsumer01 {

  private static final String TST_TOPIC_NAME = "block_img_0";

  public static void main(String[] args) throws Exception {
    new Thread(() -> {
      Properties props = new Properties();
      props.put("bootstrap.servers", "10.1.5.129:9092,10.1.5.166:9092,10.1.5.182:9092");
      props.put("group.id", "huanailiang");
//      props.put("client.id", this.consumerId);
      //  从消费获取记录的最小字节数，1的话，只要有就会取..
//      props.put("fetch.min.bytes", "1024");
//      props.put("max.poll.records", 1);
      // 最大 等待时间，和上面配置使用
      props.put("fetch.max.wait.ms", "500");
//      props.put("max.poll.interval.ms", "5000");
      // 指定了服务器从每个分区中返回给消费者的最大字节数、默认值是 1MB
      props.put("max.partition.fetch.bytes", "218");
      // 这个值非常的关键和重要、如果消费者在 sessionTimeoutMs 内没有发送心跳给服务器、会认为死亡
      props.put("session.timeout.ms", "10000");
      // 该属性指定了一个消费者在读取一个没有偏移量的分区 或者偏移量 无效的情况该作何处理.
      // earliest or latest
      props.put("auto.offset.reset", "earliest");
      props.put("enable.auto.commit", false);
      // 自动提交的时间 5s
      props.put("auto.commit.interval.ms", "5000");
      props.put("key.deserializer", "org.apache.kafka.common.serialization.StringDeserializer");
      props.put("value.deserializer", "org.apache.kafka.common.serialization.StringDeserializer");
      KafkaConsumer<String, String> consumer = new KafkaConsumer<>(props);

      consumer.subscribe(Collections.singletonList(TST_TOPIC_NAME),
          new ConsumerRebalanceListener() {
            @Override
            public void onPartitionsRevoked(final Collection<TopicPartition> partitions) {
              /*第一次*/
              System.out.println("onPartitionsRevoked");
              partitions.forEach(System.out::println);
            }

            @Override
            public void onPartitionsAssigned(final Collection<TopicPartition> partitions) {
              System.err.println("onPartitionsAssigned");
              partitions.forEach(System.err::println);
            }
          });

      try {
        while (true) {
          ConsumerRecords<String, String> consumerRecords = consumer.poll(Duration.ofSeconds(3L));
          int count = 0;
          for (ConsumerRecord<String, String> record : consumerRecords) {
            if (record == null) {
              continue;
            }
            if (record.value() == null) {
              continue;
            }
            count++;
            Thread.sleep(1000L);
            KafkaProducerDm.simpleLog("msg: %s",
                record.value());
          }
//          KafkaProducerDm.simpleError("收到了消息条数:" + count);
        }
      } catch (Exception e) {

      }


    }).start();

    System.in.read();
  }

  private int errorCnt = 0;



}
