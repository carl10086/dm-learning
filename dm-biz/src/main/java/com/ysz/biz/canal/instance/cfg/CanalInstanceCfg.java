package com.ysz.biz.canal.instance.cfg;

import com.alibaba.otter.canal.instance.manager.model.CanalParameter.HAMode;
import com.alibaba.otter.canal.instance.manager.model.CanalParameter.IndexMode;
import com.alibaba.otter.canal.instance.manager.model.CanalParameter.SourcingType;
import com.alibaba.otter.canal.protocol.position.EntryPosition;
import lombok.Getter;
import lombok.ToString;

/**
 * canal instance 配置
 */
@ToString
@Getter
public class CanalInstanceCfg {

  /**
   *  canal 默认 filter , 无、 后续可以加点数据库的
   */
  private String filter = ".*\\\\..*";

  private String dbUsername = "dbadm";
  private String dbPwd = "123456";
  private String dbAddr = "127.0.0.1:3306";
  private Boolean detectingEnable = true;                      // 是否开启心跳语句
  private Boolean heartbeatHaEnable = false;                     // 是否开启基于心跳检查的ha功能
  private String detectingSQL = "SELECT 1";                                                   // 心跳sql
  private Integer detectingIntervalInSeconds = 3;                         // 检测频率
  private Integer detectingRetryTimes = 3;                         // 心跳检查重试次数

  /**
   * canal instance 的唯一 id
   */
  private Long id = 1L;

  /**
   * instance 名称 要求唯一
   */
  private String destination = "example";

  /**
   *  目录路径、要求隔离
   */
  private String dataDir = "./conf";

  /**
   * <pre>
   *    模拟 mysql slave 需要由一个 唯一 slaveId
   *    这个 id 会由 instance destination name 唯一决定
   * </pre>
   *
   */
  private Long slaveId;

  /**
   * 数据来源类型 , 只有 Mysql
   */
  private SourcingType sourcingType = SourcingType.MYSQL;

  /**
   * HA 机制 实现
   */
  private HAMode haMode = HAMode.HEARTBEAT;
  // 网络链接参数
//  private Integer port = 11111;                     // 服务端口，独立运行时需要配置
  /**
   *
   */
  private Integer defaultConnectionTimeoutInSeconds = 30;                        // sotimeout


  /**
   * <pre>
   *  tcp  参数、这个东西设置的很不合理 ..
   *
   *  合理的姿势应该由操作系统自己控制
   * </pre>
   */
  private Integer receiveBufferSize = 64 * 1024;

  /**
   * 同上 receiveBufferSize
   */
  private Integer sendBufferSize = 64 * 1024;

  /**
   * mysql 需要的编码信息
   */
  private Byte connectionCharsetNumber = (byte) 33;
  private String connectionCharset = "UTF-8";

  /**
   * 是否开启tableMetaTSDB
   */
  private Boolean tsdbEnable = Boolean.FALSE;

  /**
   * binlog 链接信息
   */
  private IndexMode indexMode;

  private EntryPosition position;


}
