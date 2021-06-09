package com.ysz.biz.canal.instance;

import com.alibaba.otter.canal.common.alarm.CanalAlarmHandler;
import com.alibaba.otter.canal.instance.core.CanalInstance;
import com.alibaba.otter.canal.instance.core.CanalMQConfig;
import com.alibaba.otter.canal.instance.manager.CanalInstanceWithManager;
import com.alibaba.otter.canal.instance.manager.model.Canal;
import com.alibaba.otter.canal.instance.manager.model.CanalParameter;
import com.alibaba.otter.canal.instance.manager.model.CanalParameter.HAMode;
import com.alibaba.otter.canal.instance.manager.model.CanalParameter.IndexMode;
import com.alibaba.otter.canal.instance.manager.model.CanalParameter.MetaMode;
import com.alibaba.otter.canal.instance.manager.model.CanalParameter.SourcingType;
import com.alibaba.otter.canal.instance.manager.model.CanalParameter.StorageMode;
import com.alibaba.otter.canal.meta.CanalMetaManager;
import com.alibaba.otter.canal.parse.CanalEventParser;
import com.alibaba.otter.canal.protocol.ClientIdentity;
import com.alibaba.otter.canal.sink.CanalEventSink;
import com.alibaba.otter.canal.store.CanalEventStore;
import java.net.InetSocketAddress;
import java.util.Arrays;
import lombok.Getter;

public class CanalInstanceWrapper implements CanalInstance {



  /**
   * 被代理的 canal Instance 对象
   */
  @Getter
  private CanalInstanceWithManager canalInstance;

  /**
   * canal instance 内部自己的参数
   */
  private Canal canal;

  public CanalInstanceWrapper() {
//    this.canal = buildCanal();
//    this.canalInstance = new CanalInstanceWithManager(this.canal, FILTER);
  }

  public static Canal buildCanal() {
    Canal canal = new Canal();
    canal.setId(1L);
    canal.setName("example");
    canal.setDesc("随便的描述信息");

    CanalParameter parameter = new CanalParameter();

    parameter.setZkClusters(Arrays.asList("127.0.0.1:2188"));
    parameter.setMetaMode(MetaMode.MEMORY);
    parameter.setHaMode(HAMode.HEARTBEAT);
    parameter.setIndexMode(IndexMode.MEMORY);
    parameter.setStorageMode(StorageMode.MEMORY);

    parameter.setMemoryStorageRawEntry(false);
    parameter.setMemoryStorageBufferSize(32 * 1024);
    parameter.setSourcingType(SourcingType.MYSQL);
    parameter.setDbAddresses(Arrays.asList(new InetSocketAddress("10.1.4.31", 3307)));
    parameter.setDbUsername("dbadm");
    parameter.setDbPassword("123456");
//    parameter.setDbPassword("D2ZQb1wjKOTYn2U9");
    parameter
        .setPositions(Arrays.asList("{\"journalName\":\"431-bin-2.000006\",\"position\":196016L}"));
//    parameter .setPositions(Arrays.asList("{\"journalName\":\"s131-bin.002831\",\"position\":54352767L}"));

    parameter.setSlaveId(1234L);

    parameter.setDefaultConnectionTimeoutInSeconds(30);
    parameter.setConnectionCharset("UTF-8");
    parameter.setConnectionCharsetNumber((byte) 33);
    parameter.setReceiveBufferSize(8 * 1024);
    parameter.setSendBufferSize(8 * 1024);

    parameter.setBlackFilter("mysql\\.slave_.*");
    parameter.setDetectingEnable(false);
    parameter.setDetectingIntervalInSeconds(10);
    parameter.setDetectingRetryTimes(3);
    parameter.setDetectingSQL("select 1");
    canal.setCanalParameter(parameter);

    return canal;
  }

  @Override
  public String getDestination() {
    return null;
  }

  @Override
  public CanalEventParser getEventParser() {
    return null;
  }

  @Override
  public CanalEventSink getEventSink() {
    return null;
  }

  @Override
  public CanalEventStore getEventStore() {
    return null;
  }

  @Override
  public CanalMetaManager getMetaManager() {
    return null;
  }

  @Override
  public CanalAlarmHandler getAlarmHandler() {
    return null;
  }

  @Override
  public boolean subscribeChange(final ClientIdentity clientIdentity) {
    return false;
  }

  @Override
  public CanalMQConfig getMqConfig() {
    return null;
  }

  @Override
  public void start() {

  }

  @Override
  public void stop() {

  }

  @Override
  public boolean isStart() {
    return false;
  }
}
