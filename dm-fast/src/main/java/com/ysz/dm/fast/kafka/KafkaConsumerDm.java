package com.ysz.dm.fast.kafka;

public class KafkaConsumerDm {


  public static void main(String[] args) throws Exception {
    new KafkaRunner("huanailiang").start();
    System.in.read();
  }
}
