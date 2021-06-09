package com.ysz.biz.canal.connection;

import com.alibaba.otter.canal.parse.inbound.EventTransactionBuffer;
import com.alibaba.otter.canal.parse.inbound.MultiStageCoprocessor;
import com.alibaba.otter.canal.parse.inbound.mysql.MysqlMultiStageCoprocessor;
import com.alibaba.otter.canal.parse.inbound.mysql.dbsync.LogEventConvert;
import com.taobao.tddl.dbsync.binlog.LogBuffer;
import com.taobao.tddl.dbsync.binlog.LogEvent;

public class CustomProcessor extends MysqlMultiStageCoprocessor {

  private boolean running = false;

  public CustomProcessor(final int ringBufferSize, final int parserThreadCount,
      final LogEventConvert logEventConvert,
      final EventTransactionBuffer transactionBuffer,
      final String destination, final boolean filterDmlInsert, final boolean filterDmlUpdate,
      final boolean filterDmlDelete) {
    super(ringBufferSize, parserThreadCount, logEventConvert, transactionBuffer, destination,
        filterDmlInsert, filterDmlUpdate, filterDmlDelete);
  }

  @Override
  public void setBinlogChecksum(final int binlogChecksum) {
  }

  @Override
  public boolean publish(final LogBuffer buffer) {
    return true;
  }

  @Override
  public boolean publish(final LogEvent event) {
    return true;
  }

  @Override
  public void start() {
    this.running = true;
  }

  @Override
  public void stop() {
    this.running = false;
  }

  @Override
  public boolean isStart() {
    return this.running;
  }
}
