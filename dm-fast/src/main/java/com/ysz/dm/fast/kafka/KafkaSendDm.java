package com.ysz.dm.fast.kafka;

import com.ysz.dm.fast.jmh.StringTools;
import java.util.Date;
import java.util.Properties;
import java.util.concurrent.atomic.AtomicBoolean;
import org.apache.commons.lang3.time.FastDateFormat;
import org.apache.kafka.clients.producer.Callback;
import org.apache.kafka.clients.producer.KafkaProducer;
import org.apache.kafka.clients.producer.ProducerRecord;
import org.apache.kafka.clients.producer.RecordMetadata;

public class KafkaSendDm {

  private final String tstTopicName = "tsttopic";

  private KafkaProducer<String, String> kafkaProducer;

  private AtomicBoolean inited = new AtomicBoolean(false);

  private void init() {
    if (inited.compareAndSet(false, true)) {
      this.kafkaProducer = new KafkaProducer<>(initProp());
    }
  }

  public KafkaSendDm() {
    this.init();
  }

  private Properties initProp() {
    Properties props = new Properties();
    props.put("bootstrap.servers", "10.1.5.129:9092,10.1.5.166:9092,10.1.5.182:9092");
    props.put("acks", "all");
//    props.put("buffer.memory", );
    props.put("retries", "0");
//    props.put("batch.size", config.getBatchSize());
//    props.put("linger.ms", config.getLingerMs());
//    props.put("max.in.flight.requests.per.connection",
//        config.getMaxInFlightRequestsPerConnection());
    props.put("request.timeout.ms", "2000");
    props.put("max.block.ms", "2000");
    props.put("delivery.timeout.ms", "2100");
//    props.put("metadata.fetch.timeout.ms", config.getMetadataFetchTimeoutMs());
//    props.put("max.block.ms", config.getMaxBlockMs());
//    props.put("max.request.size", config.getMaxRequestSize());
//    props.put("receive.buffer.bytes", config.getReceiveBufferBytes());
//    props.put("send.buffer.bytes", config.getSendBufferBytes());
    props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
    props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer");
    props.put("serializer.class", "kafka.serializer.StringEncoder");
    return props;
  }

  private void sendSingle(int msgIndex) {
    kafkaProducer.send(new ProducerRecord<>(tstTopicName, "hello world:" + msgIndex));
    simpleLog("发送成功");
  }

  private void sendWithCallback(final int msgIndex) {
    kafkaProducer.send(new ProducerRecord<>(tstTopicName, "hello world:" + msgIndex),
        new Callback() {
          @Override
          public void onCompletion(RecordMetadata recordMetadata, Exception e) {
            if (e != null) {
              simpleError("发送失败:%s" + recordMetadata.partition(), msgIndex);
            } else {
              simpleLog("发送成功:%s," + recordMetadata.partition(), msgIndex);
            }
          }
        }
    );
  }

  private void flush() {
    kafkaProducer.flush();
  }

  private void simpleError(String format, Object... params) {
    System.err.println(now() + StringTools.formatWithSpecial(format, params));
  }

  private String now() {
    return FastDateFormat.getInstance("yyyy-MM-dd HH:mm:ss ->").format(new Date());
  }


  private void simpleLog(String format, Object... params) {
    System.out.println(now() + StringTools.formatWithSpecial(format, params));
  }


  public static void main(String[] args) throws Exception {
    KafkaSendDm kafkaSendDm = new KafkaSendDm();
    for (int i = 0; i < 10; i++) {
      kafkaSendDm.sendWithCallback(i);
      Thread.sleep(1000L);
    }
    kafkaSendDm.flush();
    System.in.read();
  }

}
