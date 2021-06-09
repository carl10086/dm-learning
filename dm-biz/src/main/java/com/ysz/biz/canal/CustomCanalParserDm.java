package com.ysz.biz.canal;

import com.alibaba.otter.canal.common.alarm.LogAlarmHandler;
import com.alibaba.otter.canal.filter.aviater.AviaterRegexFilter;
import com.alibaba.otter.canal.meta.PeriodMixedMetaManager;
import com.alibaba.otter.canal.parse.CanalHASwitchable;
import com.alibaba.otter.canal.parse.ha.CanalHAController;
import com.alibaba.otter.canal.parse.ha.HeartBeatHAController;
import com.alibaba.otter.canal.parse.inbound.mysql.MysqlEventParser;
import com.alibaba.otter.canal.parse.index.FailbackLogPositionManager;
import com.alibaba.otter.canal.parse.index.MemoryLogPositionManager;
import com.alibaba.otter.canal.parse.index.MetaLogPositionManager;
import com.alibaba.otter.canal.parse.support.AuthenticationInfo;
import java.net.InetSocketAddress;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public class CustomCanalParserDm {

  private MysqlEventParser buildParser() {
    MysqlEventParser mysqlEventParser = new MysqlEventParser();

    /* canal 中给 instance 定义的名称*/
    final String destination = "custom";
    mysqlEventParser.setDestination(destination);
    /*canal 模拟的 slaveId*/
    final Integer slaveId = 0;
    mysqlEventParser.setSlaveId(slaveId);
    /*canal 中是否开启心跳检查*/
    final Boolean detectingEnabled = false;
    mysqlEventParser.setDetectingEnable(detectingEnabled);
    /*canal 中心跳的检查语句*/
    final String detectingSql = "SELECT 1";
    mysqlEventParser.setDetectingSQL(detectingSql);
    /*开启心跳检查后的频率*/
    final Integer detectingIntervalInSeconds = 3;
    mysqlEventParser.setDefaultConnectionTimeoutInSeconds(detectingIntervalInSeconds);


    /* HA 控制器实现*/
    mysqlEventParser.setHaController(buildCanalHAController(mysqlEventParser));

    /*报警处理器*/
    mysqlEventParser.setAlarmHandler(new LogAlarmHandler());

    /*解析过滤处理器*/
    mysqlEventParser.setEventFilter(new AviaterRegexFilter(".*\\\\..*"));
    mysqlEventParser.setEventBlackFilter(new AviaterRegexFilter("mysql\\\\.slave_.*"));

    /*最大事务解析大小*/
    final Integer transactionSize = 1024;
    mysqlEventParser.setTransactionSize(transactionSize);

    /*网络参数配置 ?*/
    mysqlEventParser.setReceiveBufferSize(16384);
    mysqlEventParser.setSendBufferSize(16384);
    mysqlEventParser.setDefaultConnectionTimeoutInSeconds(30);

    /*解析编码*/
    mysqlEventParser.setConnectionCharset("UTF-8");

    /*解析位点记录*/
    FailbackLogPositionManager failbackLogPositionManager = new FailbackLogPositionManager(
        new MemoryLogPositionManager(),
        new MetaLogPositionManager(buildMetaManager())
    );
    mysqlEventParser.setLogPositionManager(failbackLogPositionManager);

    /*failover 切换时间回退的时间*/
    final int fallbackIntervalInSeconds = 60;
    mysqlEventParser.setFallbackIntervalInSeconds(fallbackIntervalInSeconds);


    /*解析数据库的信息*/
    AuthenticationInfo authenticationInfo = new AuthenticationInfo();
    authenticationInfo.setAddress(new InetSocketAddress("10.1.4.31", 3307));
    authenticationInfo.setUsername("dbadm");
    authenticationInfo.setPassword("123456");
    authenticationInfo.setDefaultDatabaseName("zcwdb");
    try {
      authenticationInfo.initPwd();
    } catch (Exception e) {
      log.error("initPwd 失败", e);
    }
    mysqlEventParser.setMasterInfo(authenticationInfo);

    mysqlEventParser.setFilterQueryDml(false);
    // ...

    // binlog 格式暂时不设置, 设置了会强制校验
    mysqlEventParser.setSupportBinlogFormats(null);
    mysqlEventParser.setSupportBinlogImages(null);

    mysqlEventParser.setEnableTsdb(false);
    mysqlEventParser.setIsGTIDMode(false);

    /*是否开启并行解析模式*/
    mysqlEventParser.setParallel(true);


    /*true: binlog 被删除之后、 自动按照最新的数据订阅*/
    mysqlEventParser.setAutoResetLatestPosMode(false);

    return mysqlEventParser;
  }

  private PeriodMixedMetaManager buildMetaManager() {
    return new PeriodMixedMetaManager();
  }

  private CanalHAController buildCanalHAController(CanalHASwitchable canalHASwitchable) {
    HeartBeatHAController heartBeatHAController = new HeartBeatHAController();
    /*默认3次*/
    heartBeatHAController.setDetectingRetryTimes(3);
    heartBeatHAController.setSwitchEnable(false);
    heartBeatHAController.setCanalHASwitchable(canalHASwitchable);
    return heartBeatHAController;
  }


  public static void main(String[] args) throws Exception {
    MysqlEventParser mysqlEventParser = new CustomCanalParserDm().buildParser();
    mysqlEventParser.start();
    System.in.read();
  }
}
