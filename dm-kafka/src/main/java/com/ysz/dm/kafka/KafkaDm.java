package com.ysz.dm.kafka;

import com.ysz.dm.kafka.producer.KafkaSenderConfig;
import com.ysz.dm.kafka.producer.MsgSenderKafkaBean;
import com.ysz.dm.kafka.producer.SendResult;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

public class KafkaDm {

  private MsgSenderKafkaBean senderKafkaBean;

  @Before
  public void setUp() throws Exception {
    KafkaSenderConfig config = new KafkaSenderConfig();
    config.setAck("0");
    config.setBootstrapServers("127.0.0.1:9092");
    config.setMaxBlockMs(1000);
    senderKafkaBean = new MsgSenderKafkaBean(config);
    senderKafkaBean.afterPropertiesSet();
  }

  @Test
  public void testKafkaProducer() {
    for (int i = 0; i < 10; i++) {
      SendResult sendResult = senderKafkaBean.sendMsg(
          "test",
          "aaa",
          500
      );
      System.out.println(sendResult);
    }
  }

  @After
  public void tearDown() throws Exception {
    senderKafkaBean.destroy();
  }
}
