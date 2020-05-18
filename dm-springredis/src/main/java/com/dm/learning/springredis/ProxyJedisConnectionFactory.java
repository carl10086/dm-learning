package com.dm.learning.springredis;

import org.apache.commons.pool2.impl.GenericObjectPoolConfig;
import org.springframework.beans.factory.DisposableBean;
import org.springframework.beans.factory.InitializingBean;
import org.springframework.dao.DataAccessException;
import org.springframework.data.redis.connection.RedisClusterConfiguration;
import org.springframework.data.redis.connection.RedisClusterConnection;
import org.springframework.data.redis.connection.RedisConnection;
import org.springframework.data.redis.connection.RedisConnectionFactory;
import org.springframework.data.redis.connection.RedisSentinelConfiguration;
import org.springframework.data.redis.connection.RedisSentinelConnection;
import org.springframework.data.redis.connection.RedisStandaloneConfiguration;
import org.springframework.data.redis.connection.jedis.JedisClientConfiguration;
import org.springframework.data.redis.connection.jedis.JedisConnectionFactory;
import org.springframework.lang.Nullable;
import redis.clients.jedis.JedisPoolConfig;
import redis.clients.jedis.JedisShardInfo;

public class ProxyJedisConnectionFactory implements InitializingBean, DisposableBean,
    RedisConnectionFactory {

  private final JedisConnectionFactory jedisConnectionFactory;
  private String endpoint;

  public ProxyJedisConnectionFactory(
      JedisConnectionFactory jedisConnectionFactory) {
    this.jedisConnectionFactory = jedisConnectionFactory;
  }

  @Override
  public void afterPropertiesSet() {
    jedisConnectionFactory.afterPropertiesSet();
  }

  @Override
  public void destroy() {
    jedisConnectionFactory.destroy();
  }

  @Override
  public RedisConnection getConnection() {
    return new ProxyRedisConnection(jedisConnectionFactory.getConnection());
  }

  @Override
  public RedisClusterConnection getClusterConnection() {
    return jedisConnectionFactory.getClusterConnection();
  }

  @Override
  public DataAccessException translateExceptionIfPossible(RuntimeException ex) {
    return jedisConnectionFactory.translateExceptionIfPossible(ex);
  }

  public String getHostName() {
    return jedisConnectionFactory.getHostName();
  }

  @Deprecated
  public void setHostName(String hostName) {
    jedisConnectionFactory.setHostName(hostName);
  }

  public boolean isUseSsl() {
    return jedisConnectionFactory.isUseSsl();
  }

  @Deprecated
  public void setUseSsl(boolean useSsl) {
    jedisConnectionFactory.setUseSsl(useSsl);
  }

  @Nullable
  public String getPassword() {
    return jedisConnectionFactory.getPassword();
  }

  @Deprecated
  public void setPassword(String password) {
    jedisConnectionFactory.setPassword(password);
  }

  public int getPort() {
    return jedisConnectionFactory.getPort();
  }

  @Deprecated
  public void setPort(int port) {
    jedisConnectionFactory.setPort(port);
  }

  @Nullable
  @Deprecated
  public JedisShardInfo getShardInfo() {
    return jedisConnectionFactory.getShardInfo();
  }

  @Deprecated
  public void setShardInfo(JedisShardInfo shardInfo) {
    jedisConnectionFactory.setShardInfo(shardInfo);
  }

  public int getTimeout() {
    return jedisConnectionFactory.getTimeout();
  }

  @Deprecated
  public void setTimeout(int timeout) {
    jedisConnectionFactory.setTimeout(timeout);
  }

  public boolean getUsePool() {
    return jedisConnectionFactory.getUsePool();
  }

  @Deprecated
  public void setUsePool(boolean usePool) {
    jedisConnectionFactory.setUsePool(usePool);
  }

  @Nullable
  public GenericObjectPoolConfig getPoolConfig() {
    return jedisConnectionFactory.getPoolConfig();
  }

  @Deprecated
  public void setPoolConfig(JedisPoolConfig poolConfig) {
    jedisConnectionFactory.setPoolConfig(poolConfig);
  }

  public int getDatabase() {
    return jedisConnectionFactory.getDatabase();
  }

  @Deprecated
  public void setDatabase(int index) {
    jedisConnectionFactory.setDatabase(index);
  }

  @Nullable
  public String getClientName() {
    return jedisConnectionFactory.getClientName();
  }

  @Deprecated
  public void setClientName(String clientName) {
    jedisConnectionFactory.setClientName(clientName);
  }

  public JedisClientConfiguration getClientConfiguration() {
    return jedisConnectionFactory.getClientConfiguration();
  }

  @Nullable
  public RedisStandaloneConfiguration getStandaloneConfiguration() {
    return jedisConnectionFactory.getStandaloneConfiguration();
  }

  @Nullable
  public RedisSentinelConfiguration getSentinelConfiguration() {
    return jedisConnectionFactory.getSentinelConfiguration();
  }

  @Nullable
  public RedisClusterConfiguration getClusterConfiguration() {
    return jedisConnectionFactory.getClusterConfiguration();
  }

  @Override
  public boolean getConvertPipelineAndTxResults() {
    return jedisConnectionFactory.getConvertPipelineAndTxResults();
  }

  public void setConvertPipelineAndTxResults(boolean convertPipelineAndTxResults) {
    jedisConnectionFactory.setConvertPipelineAndTxResults(convertPipelineAndTxResults);
  }

  public boolean isRedisSentinelAware() {
    return jedisConnectionFactory.isRedisSentinelAware();
  }

  public boolean isRedisClusterAware() {
    return jedisConnectionFactory.isRedisClusterAware();
  }

  @Override
  public RedisSentinelConnection getSentinelConnection() {
    return jedisConnectionFactory.getSentinelConnection();
  }
}
