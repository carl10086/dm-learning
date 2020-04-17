package com.ysz.dm.kafka.producer;


import com.google.common.base.Preconditions;
import java.util.Properties;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import lombok.extern.slf4j.Slf4j;
import org.apache.kafka.clients.producer.KafkaProducer;
import org.apache.kafka.clients.producer.ProducerConfig;
import org.apache.kafka.clients.producer.ProducerRecord;
import org.apache.kafka.clients.producer.RecordMetadata;

/**
 * @author carl.yu
 * @date 2018/9/17
 */
@Slf4j
public class MsgSenderKafkaBean {


  private KafkaProducer<String, String> producer;
  private final KafkaSenderConfig config;


  public MsgSenderKafkaBean(KafkaSenderConfig config) {
    this.config = config;
  }


  public SendResult sendMsg(String topic, String sendString, Integer sendTimeoutMs) {

    Future<RecordMetadata> sendFuture = producer
        .send(new ProducerRecord<>(getTopic(config.getNamespace(), topic), sendString));
    try {
      if (sendTimeoutMs <= 0L) {
        return SendResult.success(new KafkaSendResult(sendFuture.get()));
      } else {
        return SendResult
            .success(new KafkaSendResult(sendFuture.get(sendTimeoutMs, TimeUnit.MILLISECONDS)));
      }
    } catch (Exception e) {
      return SendResult.error(e, "send failed");
    }
  }

//  public void sendMsgWithCallback(String topic, DtMsg msg,
//      DtSendCallback callback) {
//    String sendString = null;
//    try {
//      sendString = objectMapper.writeValueAsString(msg);
//    } catch (JsonProcessingException e) {
//      callback.onCompletion(SendResult.error(e, "parse json failed"));
//    }
//    producer.send(new ProducerRecord<>(getTopic(config.getNamespace(), topic), sendString),
//        (recordMetadata, e) -> {
//          if (e != null) {
//            callback.onCompletion(SendResult.error(e));
//          } else {
//            callback.onCompletion(SendResult.success(new KafkaSendResult(recordMetadata)));
//          }
//        });
//  }


  private String getTopic(String ns, String topic) {
    if (ns == null || ns.length() == 0) {
      return topic;
    }
    return String.format("%s-%s", ns, topic);
  }

  public void destroy() throws Exception {
    if (producer != null) {
      this.producer.close();
    }
    log.info("MsgSenderKafkaImpl stop success:{}", this.config);
  }

  public void afterPropertiesSet() throws Exception {
    validateConfig();
    Properties props = new Properties();
    props.put("bootstrap.servers", config.getBootstrapServers());
    props.put("acks", config.getAck());
    props.put("buffer.memory", config.getBufferMemory());
    props.put("retries", config.getRetries());
    props.put("batch.size", config.getBatchSize());
    props.put("linger.ms", config.getLingerMs());
    props.put("max.in.flight.requests.per.connection",
        config.getMaxInFlightRequestsPerConnection());
    props.put("timeout.ms", config.getTimeoutMs());
    props.put("request.timeout.ms", config.getRequestTimeoutMs());
    props.put("metadata.fetch.timeout.ms", config.getMetadataFetchTimeoutMs());
    props.put(ProducerConfig.MAX_BLOCK_MS_CONFIG, config.getMaxBlockMs());
    props.put(ProducerConfig.METADATA_FETCH_TIMEOUT_CONFIG, config.getMaxBlockMs());
    props.put("max.request.size", config.getMaxRequestSize());
    props.put("receive.buffer.bytes", config.getReceiveBufferBytes());
    props.put("send.buffer.bytes", config.getSendBufferBytes());
    props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
    props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer");
    props.put("serializer.class", "kafka.serializer.StringEncoder");
    this.producer = new KafkaProducer<>(props);
    log.info("MsgSenderKafkaImpl start success:{}", this.config);
  }

  private void validateConfig() {
    Preconditions.checkNotNull(config);
    Preconditions.checkNotNull(config.getBootstrapServers());
    /*如果设置了 通用的 objectmapper 可以直接复用*/
  }
}
