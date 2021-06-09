package com.ysz.biz.canal.connection;

import com.alibaba.otter.canal.parse.driver.mysql.MysqlConnector;
import com.alibaba.otter.canal.parse.driver.mysql.packets.server.ResultSetPacket;
import com.alibaba.otter.canal.parse.exception.CanalParseException;
import com.alibaba.otter.canal.parse.inbound.mysql.MysqlConnection;
import com.alibaba.otter.canal.protocol.position.EntryPosition;
import java.io.IOException;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.UnknownHostException;
import java.nio.charset.Charset;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;
import org.springframework.util.CollectionUtils;

/**
 * 这个例子中可以去解析 mysql 的 binlog 复制协议
 */
public class MysqlConnectionDm {

  private final String binLogFileName = "431-bin-2.000006";
  private final Long binlogFilePosition = 196016L;
  // 编码信息
  protected byte connectionCharsetNumber = (byte) 33;
  protected Charset connectionCharset = Charset.forName("UTF-8");
  protected final AtomicLong receivedBinlogBytes = new AtomicLong(0L);
  protected String destination;
  private long slaveId = 100001L;
  private EntryPosition entryPosition;
  private CustomProcessor processor = new CustomProcessor(
      0,
      0,
      null
      , null, null, true, true, true
  );

  public void execute() {
    MysqlConnection mysqlConnection = null;

    try {
      mysqlConnection = getMysqlConnection();
      mysqlConnection.connect();
      EntryPosition endPosition = findEndPosition(mysqlConnection);
      System.err.println(endPosition);
      mysqlConnection.dump(endPosition.getJournalName(), endPosition.getPosition(), processor);

    } catch (Exception e) {
      e.printStackTrace();
    } finally {
      try {
        if (mysqlConnection != null) {
          mysqlConnection.disconnect();
        }
      } catch (IOException e) {
        e.printStackTrace();
      }
    }
  }

  /**
   * 查询当前的binlog位置
   */
  private EntryPosition findEndPosition(MysqlConnection mysqlConnection) {
    try {
      ResultSetPacket packet = mysqlConnection.query("show master status");
      List<String> fields = packet.getFieldValues();
      if (CollectionUtils.isEmpty(fields)) {
        throw new CanalParseException(
            "command : 'show master status' has an error! pls check. you need (at least one of) the SUPER,REPLICATION CLIENT privilege(s) for this operation");
      }
      EntryPosition endPosition = new EntryPosition(fields.get(0), Long.valueOf(fields.get(1)));
      return endPosition;
    } catch (IOException e) {
      throw new CanalParseException("command : 'show master status' has an error!", e);
    }
  }


  private MysqlConnection getMysqlConnection() {
    MysqlConnection mysqlConnection = new MysqlConnection(
        new InetSocketAddress("10.1.4.31", 3307),
        "dbadm",
        "123456",
        connectionCharsetNumber,
        ""
    );
    MysqlConnector connector = mysqlConnection.getConnector();
    connector.setSendBufferSize(16384);
    connector.setReceiveBufferSize(16384);
    connector.setSoTimeout(30 * 1000);

    mysqlConnection.setCharset(connectionCharset);
    mysqlConnection.setReceivedBinlogBytes(receivedBinlogBytes);

    if (this.slaveId <= 0) {
      this.slaveId = generateUniqueServerId();
    }

    mysqlConnection.setSlaveId(slaveId);
    return mysqlConnection;
  }


  private final long generateUniqueServerId() {
    try {
      // a=`echo $masterip|cut -d\. -f1`
      // b=`echo $masterip|cut -d\. -f2`
      // c=`echo $masterip|cut -d\. -f3`
      // d=`echo $masterip|cut -d\. -f4`
      // #server_id=`expr $a \* 256 \* 256 \* 256 + $b \* 256 \* 256 + $c
      // \* 256 + $d `
      // #server_id=$b$c$d
      // server_id=`expr $b \* 256 \* 256 + $c \* 256 + $d `
      InetAddress localHost = InetAddress.getLocalHost();
      byte[] addr = localHost.getAddress();
      int salt = (destination != null) ? destination.hashCode() : 0;
      return ((0x7f & salt) << 24) + ((0xff & (int) addr[1]) << 16) // NL
          + ((0xff & (int) addr[2]) << 8) // NL
          + (0xff & (int) addr[3]);
    } catch (UnknownHostException e) {
      throw new CanalParseException("Unknown host", e);
    }
  }


  public static void main(String[] args) throws Exception {
    new MysqlConnectionDm().execute();
  }

}
