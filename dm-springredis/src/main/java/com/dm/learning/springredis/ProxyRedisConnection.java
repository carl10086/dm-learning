package com.dm.learning.springredis;

import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Properties;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import org.springframework.dao.DataAccessException;
import org.springframework.data.geo.Circle;
import org.springframework.data.geo.Distance;
import org.springframework.data.geo.GeoResults;
import org.springframework.data.geo.Metric;
import org.springframework.data.geo.Point;
import org.springframework.data.redis.connection.BitFieldSubCommands;
import org.springframework.data.redis.connection.DataType;
import org.springframework.data.redis.connection.MessageListener;
import org.springframework.data.redis.connection.RedisConnection;
import org.springframework.data.redis.connection.RedisGeoCommands;
import org.springframework.data.redis.connection.RedisHashCommands;
import org.springframework.data.redis.connection.RedisHyperLogLogCommands;
import org.springframework.data.redis.connection.RedisKeyCommands;
import org.springframework.data.redis.connection.RedisListCommands;
import org.springframework.data.redis.connection.RedisNode;
import org.springframework.data.redis.connection.RedisPipelineException;
import org.springframework.data.redis.connection.RedisScriptingCommands;
import org.springframework.data.redis.connection.RedisSentinelConnection;
import org.springframework.data.redis.connection.RedisServerCommands;
import org.springframework.data.redis.connection.RedisSetCommands;
import org.springframework.data.redis.connection.RedisStreamCommands;
import org.springframework.data.redis.connection.RedisStringCommands;
import org.springframework.data.redis.connection.RedisZSetCommands;
import org.springframework.data.redis.connection.ReturnType;
import org.springframework.data.redis.connection.SortParameters;
import org.springframework.data.redis.connection.Subscription;
import org.springframework.data.redis.connection.ValueEncoding;
import org.springframework.data.redis.connection.stream.ByteRecord;
import org.springframework.data.redis.connection.stream.Consumer;
import org.springframework.data.redis.connection.stream.MapRecord;
import org.springframework.data.redis.connection.stream.ReadOffset;
import org.springframework.data.redis.connection.stream.RecordId;
import org.springframework.data.redis.connection.stream.StreamOffset;
import org.springframework.data.redis.connection.stream.StreamReadOptions;
import org.springframework.data.redis.core.Cursor;
import org.springframework.data.redis.core.ScanOptions;
import org.springframework.data.redis.core.types.Expiration;
import org.springframework.data.redis.core.types.RedisClientInfo;
import org.springframework.lang.Nullable;

public class ProxyRedisConnection implements RedisConnection {

  private final RedisConnection redisConnection;

  public ProxyRedisConnection(
      RedisConnection redisConnection) {
    this.redisConnection = redisConnection;
  }

  @Override
  public RedisGeoCommands geoCommands() {
    return redisConnection.geoCommands();
  }

  @Override
  public RedisHashCommands hashCommands() {
    return redisConnection.hashCommands();
  }

  @Override
  public RedisHyperLogLogCommands hyperLogLogCommands() {
    return redisConnection.hyperLogLogCommands();
  }

  @Override
  public RedisKeyCommands keyCommands() {
    return redisConnection.keyCommands();
  }

  @Override
  public RedisListCommands listCommands() {
    return redisConnection.listCommands();
  }

  @Override
  public RedisSetCommands setCommands() {
    return redisConnection.setCommands();
  }

  @Override
  public RedisScriptingCommands scriptingCommands() {
    return redisConnection.scriptingCommands();
  }

  @Override
  public RedisServerCommands serverCommands() {
    return redisConnection.serverCommands();
  }

  @Override
  public RedisStreamCommands streamCommands() {
    return redisConnection.streamCommands();
  }

  @Override
  public RedisStringCommands stringCommands() {
    return redisConnection.stringCommands();
  }

  @Override
  public RedisZSetCommands zSetCommands() {
    return redisConnection.zSetCommands();
  }

  @Override
  public void close() throws DataAccessException {
    redisConnection.close();
  }

  @Override
  public boolean isClosed() {
    return redisConnection.isClosed();
  }

  @Override
  public Object getNativeConnection() {
    return redisConnection.getNativeConnection();
  }

  @Override
  public boolean isQueueing() {
    return redisConnection.isQueueing();
  }

  @Override
  public boolean isPipelined() {
    return redisConnection.isPipelined();
  }

  @Override
  public void openPipeline() {
    redisConnection.openPipeline();
  }

  @Override
  public List<Object> closePipeline() throws RedisPipelineException {
    return redisConnection.closePipeline();
  }

  @Override
  public RedisSentinelConnection getSentinelConnection() {
    return redisConnection.getSentinelConnection();
  }

  @Override
  @Nullable
  public Object execute(String s, byte[]... bytes) {
    return redisConnection.execute(s, bytes);
  }

  @Override
  @Nullable
  public Boolean exists(byte[] key) {
    System.out.println("exists");
    return redisConnection.exists(key);
  }

  @Override
  @Nullable
  public Long exists(byte[]... bytes) {
    return redisConnection.exists(bytes);
  }

  @Override
  @Nullable
  public Long del(byte[]... bytes) {
    return redisConnection.del(bytes);
  }

  @Override
  @Nullable
  public Long unlink(byte[]... bytes) {
    return redisConnection.unlink(bytes);
  }

  @Override
  @Nullable
  public DataType type(byte[] bytes) {
    return redisConnection.type(bytes);
  }

  @Override
  @Nullable
  public Long touch(byte[]... bytes) {
    return redisConnection.touch(bytes);
  }

  @Override
  @Nullable
  public Set<byte[]> keys(byte[] bytes) {
    return redisConnection.keys(bytes);
  }

  @Override
  public Cursor<byte[]> scan(ScanOptions scanOptions) {
    return redisConnection.scan(scanOptions);
  }

  @Override
  @Nullable
  public byte[] randomKey() {
    return redisConnection.randomKey();
  }

  @Override
  public void rename(byte[] bytes, byte[] bytes1) {
    redisConnection.rename(bytes, bytes1);
  }

  @Override
  @Nullable
  public Boolean renameNX(byte[] bytes, byte[] bytes1) {
    return redisConnection.renameNX(bytes, bytes1);
  }

  @Override
  @Nullable
  public Boolean expire(byte[] bytes, long l) {
    return redisConnection.expire(bytes, l);
  }

  @Override
  @Nullable
  public Boolean pExpire(byte[] bytes, long l) {
    return redisConnection.pExpire(bytes, l);
  }

  @Override
  @Nullable
  public Boolean expireAt(byte[] bytes, long l) {
    return redisConnection.expireAt(bytes, l);
  }

  @Override
  @Nullable
  public Boolean pExpireAt(byte[] bytes, long l) {
    return redisConnection.pExpireAt(bytes, l);
  }

  @Override
  @Nullable
  public Boolean persist(byte[] bytes) {
    return redisConnection.persist(bytes);
  }

  @Override
  @Nullable
  public Boolean move(byte[] bytes, int i) {
    return redisConnection.move(bytes, i);
  }

  @Override
  @Nullable
  public Long ttl(byte[] bytes) {
    return redisConnection.ttl(bytes);
  }

  @Override
  @Nullable
  public Long ttl(byte[] bytes, TimeUnit timeUnit) {
    return redisConnection.ttl(bytes, timeUnit);
  }

  @Override
  @Nullable
  public Long pTtl(byte[] bytes) {
    return redisConnection.pTtl(bytes);
  }

  @Override
  @Nullable
  public Long pTtl(byte[] bytes, TimeUnit timeUnit) {
    return redisConnection.pTtl(bytes, timeUnit);
  }

  @Override
  @Nullable
  public List<byte[]> sort(byte[] bytes,
      SortParameters sortParameters) {
    return redisConnection.sort(bytes, sortParameters);
  }

  @Override
  @Nullable
  public Long sort(byte[] bytes,
      SortParameters sortParameters, byte[] bytes1) {
    return redisConnection.sort(bytes, sortParameters, bytes1);
  }

  @Override
  @Nullable
  public byte[] dump(byte[] bytes) {
    return redisConnection.dump(bytes);
  }

  @Override
  public void restore(byte[] key, long ttlInMillis, byte[] serializedValue) {
    redisConnection.restore(key, ttlInMillis, serializedValue);
  }

  @Override
  public void restore(byte[] bytes, long l, byte[] bytes1, boolean b) {
    redisConnection.restore(bytes, l, bytes1, b);
  }

  @Override
  @Nullable
  public ValueEncoding encodingOf(byte[] bytes) {
    return redisConnection.encodingOf(bytes);
  }

  @Override
  @Nullable
  public Duration idletime(byte[] bytes) {
    return redisConnection.idletime(bytes);
  }

  @Override
  @Nullable
  public Long refcount(byte[] bytes) {
    return redisConnection.refcount(bytes);
  }

  @Override
  @Nullable
  public byte[] get(byte[] bytes) {
    return redisConnection.get(bytes);
  }

  @Override
  @Nullable
  public byte[] getSet(byte[] bytes, byte[] bytes1) {
    /**/
    String name = "getSet";
    String type = "REDIS";
    String endpoint = "127.0.0.1:6379";
    long duration;
    String params ;
    return redisConnection.getSet(bytes, bytes1);
  }

  @Override
  @Nullable
  public List<byte[]> mGet(byte[]... bytes) {
    return redisConnection.mGet(bytes);
  }

  @Override
  @Nullable
  public Boolean set(byte[] bytes, byte[] bytes1) {
    return redisConnection.set(bytes, bytes1);
  }

  @Override
  @Nullable
  public Boolean set(byte[] bytes, byte[] bytes1,
      Expiration expiration,
      SetOption setOption) {
    return redisConnection.set(bytes, bytes1, expiration, setOption);
  }

  @Override
  @Nullable
  public Boolean setNX(byte[] bytes, byte[] bytes1) {
    return redisConnection.setNX(bytes, bytes1);
  }

  @Override
  @Nullable
  public Boolean setEx(byte[] bytes, long l, byte[] bytes1) {
    return redisConnection.setEx(bytes, l, bytes1);
  }

  @Override
  @Nullable
  public Boolean pSetEx(byte[] bytes, long l, byte[] bytes1) {
    return redisConnection.pSetEx(bytes, l, bytes1);
  }

  @Override
  @Nullable
  public Boolean mSet(Map<byte[], byte[]> map) {
    return redisConnection.mSet(map);
  }

  @Override
  @Nullable
  public Boolean mSetNX(Map<byte[], byte[]> map) {
    return redisConnection.mSetNX(map);
  }

  @Override
  @Nullable
  public Long incr(byte[] bytes) {
    return redisConnection.incr(bytes);
  }

  @Override
  @Nullable
  public Long incrBy(byte[] bytes, long l) {
    return redisConnection.incrBy(bytes, l);
  }

  @Override
  @Nullable
  public Double incrBy(byte[] bytes, double v) {
    return redisConnection.incrBy(bytes, v);
  }

  @Override
  @Nullable
  public Long decr(byte[] bytes) {
    return redisConnection.decr(bytes);
  }

  @Override
  @Nullable
  public Long decrBy(byte[] bytes, long l) {
    return redisConnection.decrBy(bytes, l);
  }

  @Override
  @Nullable
  public Long append(byte[] bytes, byte[] bytes1) {
    return redisConnection.append(bytes, bytes1);
  }

  @Override
  @Nullable
  public byte[] getRange(byte[] bytes, long l, long l1) {
    return redisConnection.getRange(bytes, l, l1);
  }

  @Override
  public void setRange(byte[] bytes, byte[] bytes1, long l) {
    redisConnection.setRange(bytes, bytes1, l);
  }

  @Override
  @Nullable
  public Boolean getBit(byte[] bytes, long l) {
    return redisConnection.getBit(bytes, l);
  }

  @Override
  @Nullable
  public Boolean setBit(byte[] bytes, long l, boolean b) {
    return redisConnection.setBit(bytes, l, b);
  }

  @Override
  @Nullable
  public Long bitCount(byte[] bytes) {
    return redisConnection.bitCount(bytes);
  }

  @Override
  @Nullable
  public Long bitCount(byte[] bytes, long l, long l1) {
    return redisConnection.bitCount(bytes, l, l1);
  }

  @Override
  @Nullable
  public List<Long> bitField(byte[] bytes,
      BitFieldSubCommands bitFieldSubCommands) {
    return redisConnection.bitField(bytes, bitFieldSubCommands);
  }

  @Override
  @Nullable
  public Long bitOp(
      BitOperation bitOperation, byte[] bytes, byte[]... bytes1) {
    return redisConnection.bitOp(bitOperation, bytes, bytes1);
  }

  @Override
  @Nullable
  public Long bitPos(byte[] key, boolean bit) {
    return redisConnection.bitPos(key, bit);
  }

  @Override
  @Nullable
  public Long bitPos(byte[] bytes, boolean b,
      org.springframework.data.domain.Range<Long> range) {
    return redisConnection.bitPos(bytes, b, range);
  }

  @Override
  @Nullable
  public Long strLen(byte[] bytes) {
    return redisConnection.strLen(bytes);
  }

  @Override
  @Nullable
  public Long rPush(byte[] bytes, byte[]... bytes1) {
    return redisConnection.rPush(bytes, bytes1);
  }

  @Override
  @Nullable
  public Long lPush(byte[] bytes, byte[]... bytes1) {
    return redisConnection.lPush(bytes, bytes1);
  }

  @Override
  @Nullable
  public Long rPushX(byte[] bytes, byte[] bytes1) {
    return redisConnection.rPushX(bytes, bytes1);
  }

  @Override
  @Nullable
  public Long lPushX(byte[] bytes, byte[] bytes1) {
    return redisConnection.lPushX(bytes, bytes1);
  }

  @Override
  @Nullable
  public Long lLen(byte[] bytes) {
    return redisConnection.lLen(bytes);
  }

  @Override
  @Nullable
  public List<byte[]> lRange(byte[] bytes, long l, long l1) {
    return redisConnection.lRange(bytes, l, l1);
  }

  @Override
  public void lTrim(byte[] bytes, long l, long l1) {
    redisConnection.lTrim(bytes, l, l1);
  }

  @Override
  @Nullable
  public byte[] lIndex(byte[] bytes, long l) {
    return redisConnection.lIndex(bytes, l);
  }

  @Override
  @Nullable
  public Long lInsert(byte[] bytes,
      Position position, byte[] bytes1, byte[] bytes2) {
    return redisConnection.lInsert(bytes, position, bytes1, bytes2);
  }

  @Override
  public void lSet(byte[] bytes, long l, byte[] bytes1) {
    redisConnection.lSet(bytes, l, bytes1);
  }

  @Override
  @Nullable
  public Long lRem(byte[] bytes, long l, byte[] bytes1) {
    return redisConnection.lRem(bytes, l, bytes1);
  }

  @Override
  @Nullable
  public byte[] lPop(byte[] bytes) {
    return redisConnection.lPop(bytes);
  }

  @Override
  @Nullable
  public byte[] rPop(byte[] bytes) {
    return redisConnection.rPop(bytes);
  }

  @Override
  @Nullable
  public List<byte[]> bLPop(int i, byte[]... bytes) {
    return redisConnection.bLPop(i, bytes);
  }

  @Override
  @Nullable
  public List<byte[]> bRPop(int i, byte[]... bytes) {
    return redisConnection.bRPop(i, bytes);
  }

  @Override
  @Nullable
  public byte[] rPopLPush(byte[] bytes, byte[] bytes1) {
    return redisConnection.rPopLPush(bytes, bytes1);
  }

  @Override
  @Nullable
  public byte[] bRPopLPush(int i, byte[] bytes, byte[] bytes1) {
    return redisConnection.bRPopLPush(i, bytes, bytes1);
  }

  @Override
  @Nullable
  public Long sAdd(byte[] bytes, byte[]... bytes1) {
    return redisConnection.sAdd(bytes, bytes1);
  }

  @Override
  @Nullable
  public Long sRem(byte[] bytes, byte[]... bytes1) {
    return redisConnection.sRem(bytes, bytes1);
  }

  @Override
  @Nullable
  public byte[] sPop(byte[] bytes) {
    return redisConnection.sPop(bytes);
  }

  @Override
  @Nullable
  public List<byte[]> sPop(byte[] bytes, long l) {
    return redisConnection.sPop(bytes, l);
  }

  @Override
  @Nullable
  public Boolean sMove(byte[] bytes, byte[] bytes1, byte[] bytes2) {
    return redisConnection.sMove(bytes, bytes1, bytes2);
  }

  @Override
  @Nullable
  public Long sCard(byte[] bytes) {
    return redisConnection.sCard(bytes);
  }

  @Override
  @Nullable
  public Boolean sIsMember(byte[] bytes, byte[] bytes1) {
    return redisConnection.sIsMember(bytes, bytes1);
  }

  @Override
  @Nullable
  public Set<byte[]> sInter(byte[]... bytes) {
    return redisConnection.sInter(bytes);
  }

  @Override
  @Nullable
  public Long sInterStore(byte[] bytes, byte[]... bytes1) {
    return redisConnection.sInterStore(bytes, bytes1);
  }

  @Override
  @Nullable
  public Set<byte[]> sUnion(byte[]... bytes) {
    return redisConnection.sUnion(bytes);
  }

  @Override
  @Nullable
  public Long sUnionStore(byte[] bytes, byte[]... bytes1) {
    return redisConnection.sUnionStore(bytes, bytes1);
  }

  @Override
  @Nullable
  public Set<byte[]> sDiff(byte[]... bytes) {
    return redisConnection.sDiff(bytes);
  }

  @Override
  @Nullable
  public Long sDiffStore(byte[] bytes, byte[]... bytes1) {
    return redisConnection.sDiffStore(bytes, bytes1);
  }

  @Override
  @Nullable
  public Set<byte[]> sMembers(byte[] bytes) {
    return redisConnection.sMembers(bytes);
  }

  @Override
  @Nullable
  public byte[] sRandMember(byte[] bytes) {
    return redisConnection.sRandMember(bytes);
  }

  @Override
  @Nullable
  public List<byte[]> sRandMember(byte[] bytes, long l) {
    return redisConnection.sRandMember(bytes, l);
  }

  @Override
  public Cursor<byte[]> sScan(byte[] bytes,
      ScanOptions scanOptions) {
    return redisConnection.sScan(bytes, scanOptions);
  }

  @Override
  @Nullable
  public Boolean zAdd(byte[] bytes, double v, byte[] bytes1) {
    return redisConnection.zAdd(bytes, v, bytes1);
  }

  @Override
  @Nullable
  public Long zAdd(byte[] bytes,
      Set<Tuple> set) {
    return redisConnection.zAdd(bytes, set);
  }

  @Override
  @Nullable
  public Long zRem(byte[] bytes, byte[]... bytes1) {
    return redisConnection.zRem(bytes, bytes1);
  }

  @Override
  @Nullable
  public Double zIncrBy(byte[] bytes, double v, byte[] bytes1) {
    return redisConnection.zIncrBy(bytes, v, bytes1);
  }

  @Override
  @Nullable
  public Long zRank(byte[] bytes, byte[] bytes1) {
    return redisConnection.zRank(bytes, bytes1);
  }

  @Override
  @Nullable
  public Long zRevRank(byte[] bytes, byte[] bytes1) {
    return redisConnection.zRevRank(bytes, bytes1);
  }

  @Override
  @Nullable
  public Set<byte[]> zRange(byte[] bytes, long l, long l1) {
    return redisConnection.zRange(bytes, l, l1);
  }

  @Override
  @Nullable
  public Set<Tuple> zRangeWithScores(byte[] bytes, long l, long l1) {
    return redisConnection.zRangeWithScores(bytes, l, l1);
  }

  @Override
  @Nullable
  public Set<byte[]> zRangeByScore(byte[] key, double min, double max) {
    return redisConnection.zRangeByScore(key, min, max);
  }

  @Override
  @Nullable
  public Set<Tuple> zRangeByScoreWithScores(byte[] key,
      Range range) {
    return redisConnection.zRangeByScoreWithScores(key, range);
  }

  @Override
  @Nullable
  public Set<Tuple> zRangeByScoreWithScores(byte[] key, double min, double max) {
    return redisConnection.zRangeByScoreWithScores(key, min, max);
  }

  @Override
  @Nullable
  public Set<byte[]> zRangeByScore(byte[] key, double min, double max, long offset, long count) {
    return redisConnection.zRangeByScore(key, min, max, offset, count);
  }

  @Override
  @Nullable
  public Set<Tuple> zRangeByScoreWithScores(byte[] key, double min, double max, long offset,
      long count) {
    return redisConnection.zRangeByScoreWithScores(key, min, max, offset, count);
  }

  @Override
  @Nullable
  public Set<Tuple> zRangeByScoreWithScores(byte[] bytes,
      Range range, Limit limit) {
    return redisConnection.zRangeByScoreWithScores(bytes, range, limit);
  }

  @Override
  @Nullable
  public Set<byte[]> zRevRange(byte[] bytes, long l, long l1) {
    return redisConnection.zRevRange(bytes, l, l1);
  }

  @Override
  @Nullable
  public Set<Tuple> zRevRangeWithScores(byte[] bytes, long l, long l1) {
    return redisConnection.zRevRangeWithScores(bytes, l, l1);
  }

  @Override
  @Nullable
  public Set<byte[]> zRevRangeByScore(byte[] key, double min, double max) {
    return redisConnection.zRevRangeByScore(key, min, max);
  }

  @Override
  @Nullable
  public Set<byte[]> zRevRangeByScore(byte[] key,
      Range range) {
    return redisConnection.zRevRangeByScore(key, range);
  }

  @Override
  @Nullable
  public Set<Tuple> zRevRangeByScoreWithScores(byte[] key, double min, double max) {
    return redisConnection.zRevRangeByScoreWithScores(key, min, max);
  }

  @Override
  @Nullable
  public Set<byte[]> zRevRangeByScore(byte[] key, double min, double max, long offset, long count) {
    return redisConnection.zRevRangeByScore(key, min, max, offset, count);
  }

  @Override
  @Nullable
  public Set<byte[]> zRevRangeByScore(byte[] bytes,
      Range range, Limit limit) {
    return redisConnection.zRevRangeByScore(bytes, range, limit);
  }

  @Override
  @Nullable
  public Set<Tuple> zRevRangeByScoreWithScores(byte[] key, double min, double max, long offset,
      long count) {
    return redisConnection.zRevRangeByScoreWithScores(key, min, max, offset, count);
  }

  @Override
  @Nullable
  public Set<Tuple> zRevRangeByScoreWithScores(byte[] key,
      Range range) {
    return redisConnection.zRevRangeByScoreWithScores(key, range);
  }

  @Override
  @Nullable
  public Set<Tuple> zRevRangeByScoreWithScores(byte[] bytes,
      Range range, Limit limit) {
    return redisConnection.zRevRangeByScoreWithScores(bytes, range, limit);
  }

  @Override
  @Nullable
  public Long zCount(byte[] key, double min, double max) {
    return redisConnection.zCount(key, min, max);
  }

  @Override
  @Nullable
  public Long zCount(byte[] bytes,
      Range range) {
    return redisConnection.zCount(bytes, range);
  }

  @Override
  @Nullable
  public Long zCard(byte[] bytes) {
    return redisConnection.zCard(bytes);
  }

  @Override
  @Nullable
  public Double zScore(byte[] bytes, byte[] bytes1) {
    return redisConnection.zScore(bytes, bytes1);
  }

  @Override
  @Nullable
  public Long zRemRange(byte[] bytes, long l, long l1) {
    return redisConnection.zRemRange(bytes, l, l1);
  }

  @Override
  @Nullable
  public Long zRemRangeByScore(byte[] key, double min, double max) {
    return redisConnection.zRemRangeByScore(key, min, max);
  }

  @Override
  @Nullable
  public Long zRemRangeByScore(byte[] bytes,
      Range range) {
    return redisConnection.zRemRangeByScore(bytes, range);
  }

  @Override
  @Nullable
  public Long zUnionStore(byte[] bytes, byte[]... bytes1) {
    return redisConnection.zUnionStore(bytes, bytes1);
  }

  @Override
  @Nullable
  public Long zUnionStore(byte[] destKey,
      Aggregate aggregate, int[] weights, byte[]... sets) {
    return redisConnection.zUnionStore(destKey, aggregate, weights, sets);
  }

  @Override
  @Nullable
  public Long zUnionStore(byte[] bytes,
      Aggregate aggregate,
      Weights weights, byte[]... bytes1) {
    return redisConnection.zUnionStore(bytes, aggregate, weights, bytes1);
  }

  @Override
  @Nullable
  public Long zInterStore(byte[] bytes, byte[]... bytes1) {
    return redisConnection.zInterStore(bytes, bytes1);
  }

  @Override
  @Nullable
  public Long zInterStore(byte[] destKey,
      Aggregate aggregate, int[] weights, byte[]... sets) {
    return redisConnection.zInterStore(destKey, aggregate, weights, sets);
  }

  @Override
  @Nullable
  public Long zInterStore(byte[] bytes,
      Aggregate aggregate,
      Weights weights, byte[]... bytes1) {
    return redisConnection.zInterStore(bytes, aggregate, weights, bytes1);
  }

  @Override
  public Cursor<Tuple> zScan(byte[] bytes,
      ScanOptions scanOptions) {
    return redisConnection.zScan(bytes, scanOptions);
  }

  @Override
  @Nullable
  public Set<byte[]> zRangeByScore(byte[] key, String min, String max) {
    return redisConnection.zRangeByScore(key, min, max);
  }

  @Override
  @Nullable
  public Set<byte[]> zRangeByScore(byte[] key,
      Range range) {
    return redisConnection.zRangeByScore(key, range);
  }

  @Override
  @Nullable
  public Set<byte[]> zRangeByScore(byte[] bytes, String s, String s1, long l, long l1) {
    return redisConnection.zRangeByScore(bytes, s, s1, l, l1);
  }

  @Override
  @Nullable
  public Set<byte[]> zRangeByScore(byte[] bytes,
      Range range, Limit limit) {
    return redisConnection.zRangeByScore(bytes, range, limit);
  }

  @Override
  @Nullable
  public Set<byte[]> zRangeByLex(byte[] key) {
    return redisConnection.zRangeByLex(key);
  }

  @Override
  @Nullable
  public Set<byte[]> zRangeByLex(byte[] key,
      Range range) {
    return redisConnection.zRangeByLex(key, range);
  }

  @Override
  @Nullable
  public Set<byte[]> zRangeByLex(byte[] bytes,
      Range range, Limit limit) {
    return redisConnection.zRangeByLex(bytes, range, limit);
  }

  @Override
  @Nullable
  public Boolean hSet(byte[] bytes, byte[] bytes1, byte[] bytes2) {
    return redisConnection.hSet(bytes, bytes1, bytes2);
  }

  @Override
  @Nullable
  public Boolean hSetNX(byte[] bytes, byte[] bytes1, byte[] bytes2) {
    return redisConnection.hSetNX(bytes, bytes1, bytes2);
  }

  @Override
  @Nullable
  public byte[] hGet(byte[] bytes, byte[] bytes1) {
    return redisConnection.hGet(bytes, bytes1);
  }

  @Override
  @Nullable
  public List<byte[]> hMGet(byte[] bytes, byte[]... bytes1) {
    return redisConnection.hMGet(bytes, bytes1);
  }

  @Override
  public void hMSet(byte[] bytes, Map<byte[], byte[]> map) {
    redisConnection.hMSet(bytes, map);
  }

  @Override
  @Nullable
  public Long hIncrBy(byte[] bytes, byte[] bytes1, long l) {
    return redisConnection.hIncrBy(bytes, bytes1, l);
  }

  @Override
  @Nullable
  public Double hIncrBy(byte[] bytes, byte[] bytes1, double v) {
    return redisConnection.hIncrBy(bytes, bytes1, v);
  }

  @Override
  @Nullable
  public Boolean hExists(byte[] bytes, byte[] bytes1) {
    return redisConnection.hExists(bytes, bytes1);
  }

  @Override
  @Nullable
  public Long hDel(byte[] bytes, byte[]... bytes1) {
    return redisConnection.hDel(bytes, bytes1);
  }

  @Override
  @Nullable
  public Long hLen(byte[] bytes) {
    return redisConnection.hLen(bytes);
  }

  @Override
  @Nullable
  public Set<byte[]> hKeys(byte[] bytes) {
    return redisConnection.hKeys(bytes);
  }

  @Override
  @Nullable
  public List<byte[]> hVals(byte[] bytes) {
    return redisConnection.hVals(bytes);
  }

  @Override
  @Nullable
  public Map<byte[], byte[]> hGetAll(byte[] bytes) {
    return redisConnection.hGetAll(bytes);
  }

  @Override
  public Cursor<Entry<byte[], byte[]>> hScan(byte[] bytes,
      ScanOptions scanOptions) {
    return redisConnection.hScan(bytes, scanOptions);
  }

  @Override
  @Nullable
  public Long hStrLen(byte[] bytes, byte[] bytes1) {
    return redisConnection.hStrLen(bytes, bytes1);
  }

  @Override
  public void multi() {
    redisConnection.multi();
  }

  @Override
  public List<Object> exec() {
    return redisConnection.exec();
  }

  @Override
  public void discard() {
    redisConnection.discard();
  }

  @Override
  public void watch(byte[]... bytes) {
    redisConnection.watch(bytes);
  }

  @Override
  public void unwatch() {
    redisConnection.unwatch();
  }

  @Override
  public boolean isSubscribed() {
    return redisConnection.isSubscribed();
  }

  @Override
  @Nullable
  public Subscription getSubscription() {
    return redisConnection.getSubscription();
  }

  @Override
  @Nullable
  public Long publish(byte[] bytes, byte[] bytes1) {
    return redisConnection.publish(bytes, bytes1);
  }

  @Override
  public void subscribe(MessageListener messageListener, byte[]... bytes) {
    redisConnection.subscribe(messageListener, bytes);
  }

  @Override
  public void pSubscribe(MessageListener messageListener, byte[]... bytes) {
    redisConnection.pSubscribe(messageListener, bytes);
  }

  @Override
  public void select(int i) {
    redisConnection.select(i);
  }

  @Override
  @Nullable
  public byte[] echo(byte[] bytes) {
    return redisConnection.echo(bytes);
  }

  @Override
  @Nullable
  public String ping() {
    return redisConnection.ping();
  }

  @Override
  @Deprecated
  public void bgWriteAof() {
    redisConnection.bgWriteAof();
  }

  @Override
  public void bgReWriteAof() {
    redisConnection.bgReWriteAof();
  }

  @Override
  public void bgSave() {
    redisConnection.bgSave();
  }

  @Override
  @Nullable
  public Long lastSave() {
    return redisConnection.lastSave();
  }

  @Override
  public void save() {
    redisConnection.save();
  }

  @Override
  @Nullable
  public Long dbSize() {
    return redisConnection.dbSize();
  }

  @Override
  public void flushDb() {
    redisConnection.flushDb();
  }

  @Override
  public void flushAll() {
    redisConnection.flushAll();
  }

  @Override
  @Nullable
  public Properties info() {
    return redisConnection.info();
  }

  @Override
  @Nullable
  public Properties info(String s) {
    return redisConnection.info(s);
  }

  @Override
  public void shutdown() {
    redisConnection.shutdown();
  }

  @Override
  public void shutdown(
      ShutdownOption shutdownOption) {
    redisConnection.shutdown(shutdownOption);
  }

  @Override
  @Nullable
  public Properties getConfig(String s) {
    return redisConnection.getConfig(s);
  }

  @Override
  public void setConfig(String s, String s1) {
    redisConnection.setConfig(s, s1);
  }

  @Override
  public void resetConfigStats() {
    redisConnection.resetConfigStats();
  }

  @Override
  @Nullable
  public Long time() {
    return redisConnection.time();
  }

  @Override
  public void killClient(String s, int i) {
    redisConnection.killClient(s, i);
  }

  @Override
  public void setClientName(byte[] bytes) {
    redisConnection.setClientName(bytes);
  }

  @Override
  @Nullable
  public String getClientName() {
    return redisConnection.getClientName();
  }

  @Override
  @Nullable
  public List<RedisClientInfo> getClientList() {
    return redisConnection.getClientList();
  }

  @Override
  public void slaveOf(String s, int i) {
    redisConnection.slaveOf(s, i);
  }

  @Override
  public void slaveOfNoOne() {
    redisConnection.slaveOfNoOne();
  }

  @Override
  public void migrate(byte[] bytes, RedisNode redisNode, int i,
      MigrateOption migrateOption) {
    redisConnection.migrate(bytes, redisNode, i, migrateOption);
  }

  @Override
  public void migrate(byte[] bytes, RedisNode redisNode, int i,
      MigrateOption migrateOption, long l) {
    redisConnection.migrate(bytes, redisNode, i, migrateOption, l);
  }

  @Override
  @Nullable
  public Long xAck(byte[] key, String group, String... recordIds) {
    return redisConnection.xAck(key, group, recordIds);
  }

  @Override
  @Nullable
  public Long xAck(byte[] bytes, String s,
      RecordId... recordIds) {
    return redisConnection.xAck(bytes, s, recordIds);
  }

  @Override
  @Nullable
  public RecordId xAdd(byte[] key, Map<byte[], byte[]> content) {
    return redisConnection.xAdd(key, content);
  }

  @Override
  public RecordId xAdd(
      MapRecord<byte[], byte[], byte[]> mapRecord) {
    return redisConnection.xAdd(mapRecord);
  }

  @Override
  @Nullable
  public Long xDel(byte[] key, String... recordIds) {
    return redisConnection.xDel(key, recordIds);
  }

  @Override
  public Long xDel(byte[] bytes,
      RecordId... recordIds) {
    return redisConnection.xDel(bytes, recordIds);
  }

  @Override
  @Nullable
  public String xGroupCreate(byte[] bytes, String s,
      ReadOffset readOffset) {
    return redisConnection.xGroupCreate(bytes, s, readOffset);
  }

  @Override
  @Nullable
  public Boolean xGroupDelConsumer(byte[] key, String groupName, String consumerName) {
    return redisConnection.xGroupDelConsumer(key, groupName, consumerName);
  }

  @Override
  @Nullable
  public Boolean xGroupDelConsumer(byte[] bytes,
      Consumer consumer) {
    return redisConnection.xGroupDelConsumer(bytes, consumer);
  }

  @Override
  @Nullable
  public Boolean xGroupDestroy(byte[] bytes, String s) {
    return redisConnection.xGroupDestroy(bytes, s);
  }

  @Override
  @Nullable
  public Long xLen(byte[] bytes) {
    return redisConnection.xLen(bytes);
  }

  @Override
  @Nullable
  public List<ByteRecord> xRange(byte[] key,
      org.springframework.data.domain.Range<String> range) {
    return redisConnection.xRange(key, range);
  }

  @Override
  @Nullable
  public List<ByteRecord> xRange(byte[] bytes,
      org.springframework.data.domain.Range<String> range,
      Limit limit) {
    return redisConnection.xRange(bytes, range, limit);
  }

  @Override
  @Nullable
  public List<ByteRecord> xRead(
      StreamOffset<byte[]>... streams) {
    return redisConnection.xRead(streams);
  }

  @Override
  @Nullable
  public List<ByteRecord> xRead(
      StreamReadOptions streamReadOptions,
      StreamOffset<byte[]>... streamOffsets) {
    return redisConnection.xRead(streamReadOptions, streamOffsets);
  }

  @Override
  @Nullable
  public List<ByteRecord> xReadGroup(
      Consumer consumer,
      StreamOffset<byte[]>... streams) {
    return redisConnection.xReadGroup(consumer, streams);
  }

  @Override
  @Nullable
  public List<ByteRecord> xReadGroup(
      Consumer consumer,
      StreamReadOptions streamReadOptions,
      StreamOffset<byte[]>... streamOffsets) {
    return redisConnection.xReadGroup(consumer, streamReadOptions, streamOffsets);
  }

  @Override
  @Nullable
  public List<ByteRecord> xRevRange(byte[] key,
      org.springframework.data.domain.Range<String> range) {
    return redisConnection.xRevRange(key, range);
  }

  @Override
  @Nullable
  public List<ByteRecord> xRevRange(byte[] bytes,
      org.springframework.data.domain.Range<String> range,
      Limit limit) {
    return redisConnection.xRevRange(bytes, range, limit);
  }

  @Override
  @Nullable
  public Long xTrim(byte[] bytes, long l) {
    return redisConnection.xTrim(bytes, l);
  }

  @Override
  public void scriptFlush() {
    redisConnection.scriptFlush();
  }

  @Override
  public void scriptKill() {
    redisConnection.scriptKill();
  }

  @Override
  @Nullable
  public String scriptLoad(byte[] bytes) {
    return redisConnection.scriptLoad(bytes);
  }

  @Override
  @Nullable
  public List<Boolean> scriptExists(String... strings) {
    return redisConnection.scriptExists(strings);
  }

  @Override
  @Nullable
  public <T> T eval(byte[] bytes, ReturnType returnType, int i, byte[]... bytes1) {
    return redisConnection.eval(bytes, returnType, i, bytes1);
  }

  @Override
  @Nullable
  public <T> T evalSha(String s, ReturnType returnType, int i, byte[]... bytes) {
    return redisConnection.evalSha(s, returnType, i, bytes);
  }

  @Override
  @Nullable
  public <T> T evalSha(byte[] bytes, ReturnType returnType, int i, byte[]... bytes1) {
    return redisConnection.evalSha(bytes, returnType, i, bytes1);
  }

  @Override
  @Nullable
  public Long geoAdd(byte[] bytes, Point point, byte[] bytes1) {
    return redisConnection.geoAdd(bytes, point, bytes1);
  }

  @Override
  @Nullable
  public Long geoAdd(byte[] key,
      GeoLocation<byte[]> location) {
    return redisConnection.geoAdd(key, location);
  }

  @Override
  @Nullable
  public Long geoAdd(byte[] bytes, Map<byte[], Point> map) {
    return redisConnection.geoAdd(bytes, map);
  }

  @Override
  @Nullable
  public Long geoAdd(byte[] bytes,
      Iterable<GeoLocation<byte[]>> iterable) {
    return redisConnection.geoAdd(bytes, iterable);
  }

  @Override
  @Nullable
  public Distance geoDist(byte[] bytes, byte[] bytes1, byte[] bytes2) {
    return redisConnection.geoDist(bytes, bytes1, bytes2);
  }

  @Override
  @Nullable
  public Distance geoDist(byte[] bytes, byte[] bytes1, byte[] bytes2,
      Metric metric) {
    return redisConnection.geoDist(bytes, bytes1, bytes2, metric);
  }

  @Override
  @Nullable
  public List<String> geoHash(byte[] bytes, byte[]... bytes1) {
    return redisConnection.geoHash(bytes, bytes1);
  }

  @Override
  @Nullable
  public List<Point> geoPos(byte[] bytes, byte[]... bytes1) {
    return redisConnection.geoPos(bytes, bytes1);
  }

  @Override
  @Nullable
  public GeoResults<GeoLocation<byte[]>> geoRadius(byte[] bytes,
      Circle circle) {
    return redisConnection.geoRadius(bytes, circle);
  }

  @Override
  @Nullable
  public GeoResults<GeoLocation<byte[]>> geoRadius(byte[] bytes,
      Circle circle,
      GeoRadiusCommandArgs geoRadiusCommandArgs) {
    return redisConnection.geoRadius(bytes, circle, geoRadiusCommandArgs);
  }

  @Override
  @Nullable
  public GeoResults<GeoLocation<byte[]>> geoRadiusByMember(byte[] key, byte[] member,
      double radius) {
    return redisConnection.geoRadiusByMember(key, member, radius);
  }

  @Override
  @Nullable
  public GeoResults<GeoLocation<byte[]>> geoRadiusByMember(byte[] bytes, byte[] bytes1,
      Distance distance) {
    return redisConnection.geoRadiusByMember(bytes, bytes1, distance);
  }

  @Override
  @Nullable
  public GeoResults<GeoLocation<byte[]>> geoRadiusByMember(byte[] bytes, byte[] bytes1,
      Distance distance,
      GeoRadiusCommandArgs geoRadiusCommandArgs) {
    return redisConnection.geoRadiusByMember(bytes, bytes1, distance, geoRadiusCommandArgs);
  }

  @Override
  @Nullable
  public Long geoRemove(byte[] bytes, byte[]... bytes1) {
    return redisConnection.geoRemove(bytes, bytes1);
  }

  @Override
  @Nullable
  public Long pfAdd(byte[] bytes, byte[]... bytes1) {
    return redisConnection.pfAdd(bytes, bytes1);
  }

  @Override
  @Nullable
  public Long pfCount(byte[]... bytes) {
    return redisConnection.pfCount(bytes);
  }

  @Override
  public void pfMerge(byte[] bytes, byte[]... bytes1) {
    redisConnection.pfMerge(bytes, bytes1);
  }
}
