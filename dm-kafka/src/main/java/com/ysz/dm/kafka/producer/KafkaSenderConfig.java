package com.ysz.dm.kafka.producer;

import lombok.Data;

/**
 * @author carl.yu
 * @date 2018/9/17
 */
@Data
public class KafkaSenderConfig {

  /**
   * kafka 的 server 信息
   */
//  private String bootstrapServers = "a-1.kafka.infra.duitang.net:9092,a-2.kafka.infra.duitang.net:9092,a-3.kafka.infra.duitang.net:9092";
  private String bootstrapServers;

  /**
   * 1 表示至少 master 能够收到
   */
  private String ack = "1";

  /**
   * 0 表示不会有任何的重试
   */
  private Integer retries = 0;

  /**
   * 当多个消息发送到同一个分区的时候、 生产者会放在同一个批次里面, 这个参数指定了一个批次的大小
   */
  private Integer batchSize = 16384;

  /**
   * nagle 算法指定了发送批次之前会等待更多的消息加入批次的时间, 单位毫秒
   */
  private Integer lingerMs = 0;

  /**
   * 发送的内存缓存区, 单位字节，如果达到缓冲还有数据，取决于 max.block.ms 决定行为
   */
  private Integer bufferMemory = 33554432;

  /**
   * broker 等待同步副本返回消息确认的时间
   */
  private Integer timeoutMs = 30000;

  /**
   * 指定了生产者在发送数据时 等待服务器 返回响应的时间
   */
  private Integer requestTimeoutMs = 10000;

  /**
   * 指定了 获取元数据的时候等待服务器返回响应的时间
   */
  private Integer metadataFetchTimeoutMs = 60000;

  /**
   * 当发送缓冲慢的时候 或者没有可用的元数据的时候、 阻塞的时候到达 max.block.ms 的时候会出发 超时异常
   */
  private Integer maxBlockMs = 60000;

  /**
   * 能飞的消息个数
   */
  private Integer maxInFlightRequestsPerConnection = 10;


  /**
   * 生产者发送请求的大小、 可以指定发送的单个消息的最大值、也可以指定单个请求里的所有消息的综合.
   */
  private Integer maxRequestSize = 1048576;

  /**
   * TCP 接收数据报缓冲大小
   */
  private Integer receiveBufferBytes = 32768;

  /**
   * TCP 发送数据报缓冲大小
   */
  private Integer sendBufferBytes = 131072;

  private String namespace;
}
