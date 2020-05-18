package com.dm.learning.springredis;

import org.springframework.data.redis.connection.jedis.JedisConnectionFactory;
import org.springframework.data.redis.core.StringRedisTemplate;
import redis.clients.jedis.JedisPoolConfig;

public class JedisUtils {

  private static JedisConnectionFactory getJedisConnectionFactory(String host, int port,
      int cacheDb) {
    JedisPoolConfig config = createConfig(30, 10, 500, true, false);
    config.setTimeBetweenEvictionRunsMillis(0L);
    JedisConnectionFactory factory =
        createFactory(host, port, cacheDb, true, config);
    factory.setTimeout(10);
    return factory;
  }

  public static JedisPoolConfig createConfig(int maxTotal, int maxIdle, int maxWaitMillis,
      boolean testOnBorrow, boolean testWhileIdle) {
    JedisPoolConfig config = new JedisPoolConfig();
    config.setMaxTotal(maxTotal);
    config.setMaxIdle(maxIdle);
    config.setMaxWaitMillis((long) maxWaitMillis);
    config.setTestOnBorrow(testOnBorrow);
    config.setTestWhileIdle(testWhileIdle);
    return config;
  }

  public static JedisConnectionFactory createFactory(String host, int port, int db, boolean usePool,
      JedisPoolConfig config) {
    JedisConnectionFactory fact = new JedisConnectionFactory(config);
    fact.setUsePool(usePool);
    fact.setHostName(host);
    fact.setPort(port);
    fact.setDatabase(db);
    fact.afterPropertiesSet();
    return fact;
  }


  public static StringRedisTemplate redisCacheTemplate(String redisCacheHost,
      Integer redisCachePort) {
    JedisConnectionFactory factory =
        getJedisConnectionFactory(redisCacheHost, redisCachePort, 0);
    return new StringRedisTemplate(new ProxyJedisConnectionFactory(
        factory
    ));
  }
}
