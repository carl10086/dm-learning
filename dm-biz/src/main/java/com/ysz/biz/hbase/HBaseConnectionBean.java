package com.ysz.biz.hbase;

import lombok.Getter;
import lombok.Setter;
import org.apache.commons.lang3.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.hbase.HBaseConfiguration;
import org.apache.hadoop.hbase.client.Connection;
import org.apache.hadoop.hbase.client.ConnectionFactory;
import org.springframework.beans.factory.DisposableBean;
import org.springframework.beans.factory.InitializingBean;

/**
 * @author carl
 */
public class HBaseConnectionBean implements InitializingBean, DisposableBean {

  private Configuration configuration;
  @Getter
  private Connection connection;

  @Getter
  @Setter
  private String hbaseZk;

  @Override
  public void destroy() throws Exception {
    if (connection != null) {
      connection.close();
    }
  }

  @Override
  public void afterPropertiesSet() throws Exception {
    configuration = HBaseConfiguration.create();
//    configuration.set("hbase.zookeeper.quorum", "10.1.13.10,10.1.13.11,10.1.13.12");
    configuration.set("hbase.zookeeper.quorum", hbaseZk);
    connection = ConnectionFactory.createConnection(configuration);
  }


  /**
   * 10_000_000_000
   * 11 位的字符串，可以表示 100亿 -1000亿的用户级别, 目测够用
   */
  private static final int UID_FIX_LENGTH = 11;


  /**
   * 策略：定长字符串、然后反转
   * @param uid 用户id
   * @return
   */
  public static String normalize(String uid) {
    return StringUtils.reverse(fillZeroPrefix(uid));
  }


  private static String fillZeroPrefix(String uid) {
    if (uid.length() > UID_FIX_LENGTH) {
      throw new IllegalArgumentException("长度超过了11位:" + uid.length());
    }
    if (uid.length() == UID_FIX_LENGTH) {
      return uid;
    }
    int prefixLen = UID_FIX_LENGTH - uid.length();
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < prefixLen; i++) {
      sb.append("0");
    }
    return sb.toString() + uid;
  }

}
