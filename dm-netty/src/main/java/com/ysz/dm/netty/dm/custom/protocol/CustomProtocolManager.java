package com.ysz.dm.netty.dm.custom.protocol;

import com.google.common.base.Preconditions;
import com.ysz.dm.netty.dm.custom.protocol.constants.CustomProtocolCodeType;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * 这里抄的是 sofa 的代码；
 *
 * 感觉设计上. volatile  = jdk8 以上 ConcurrentMap> 读写锁 > jdk8 以下ConcurrentMap
 */
public class CustomProtocolManager {

  private Map<CustomProtocolCodeType, CustomProtocol> protocols;
  private ReadWriteLock lock;
  private Lock readLock;
  private Lock writeLock;

  private CustomProtocolManager() {
    initLock();
    /*8 个协议感觉比默认的 16 更合理一些*/
    this.protocols = new HashMap<>(8);
  }

  public static CustomProtocolManager getInstance() {
    return SingletonHolder.instance;
  }

  public void register(CustomProtocol protocol) {
    Preconditions.checkNotNull(protocol);
    final CustomProtocolCodeType code = protocol.code();
    Preconditions.checkNotNull(code);

    readLock.lock();
    try {
      Preconditions
          .checkState(!protocols.containsKey(code), "protocol code must be unique, {}", code);
    } finally {
      readLock.unlock();
    }

    writeLock.lock();
    try {
      Preconditions
          .checkState(!protocols.containsKey(code), "protocol code must be unique, {}", code);
      this.protocols.put(code, protocol);
    } finally {
      writeLock.unlock();
    }
  }


  private void initLock() {
    this.lock = new ReentrantReadWriteLock(false);
    this.readLock = lock.readLock();
    this.writeLock = lock.writeLock();
  }

  public CustomProtocol findProtocol(final CustomProtocolCodeType protocolCodeType) {
    readLock.lock();
    try {
      return protocols.get(protocolCodeType);
    } finally {
      readLock.unlock();
    }
  }


  private static class SingletonHolder {

    private static CustomProtocolManager instance = new CustomProtocolManager();
  }

}
