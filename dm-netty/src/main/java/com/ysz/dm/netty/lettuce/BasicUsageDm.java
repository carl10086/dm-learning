package com.ysz.dm.netty.lettuce;

import io.lettuce.core.RedisClient;
import io.lettuce.core.api.StatefulRedisConnection;
import io.lettuce.core.api.sync.RedisCommands;

public class BasicUsageDm {

  public static void main(String[] args) throws Exception {
    RedisClient redisClient = RedisClient.create("redis://10.1.4.31:6379/0");
    /*线程安全、也就是可以单例的意思*/
    final StatefulRedisConnection<String, String> connection = redisClient.connect();
    RedisCommands<String, String> syncCommands = connection.sync();
    final String info = syncCommands.info();
    connection.close();
    redisClient.shutdown();
    System.out.println(info);
  }

}
