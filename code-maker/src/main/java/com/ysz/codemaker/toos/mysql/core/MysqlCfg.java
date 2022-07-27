package com.ysz.codemaker.toos.mysql.core;

import java.sql.Connection;
import java.sql.DatabaseMetaData;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import lombok.Getter;
import lombok.ToString;
import lombok.extern.slf4j.Slf4j;

/**
 * 如何获取到指定 column 元数据信息. 参考: https://www.progress.com/blogs/jdbc-tutorial-extracting-database-metadata-via-jdbc-driver
 */
@ToString
@Getter
@Slf4j
public class MysqlCfg {

  static {
    try {
      Class.forName("com.mysql.cj.jdbc.Driver");
    } catch (ClassNotFoundException e) {
      throw new RuntimeException(e);
    }
  }

  private String jdbcUrl;
  private String username;
  private String password;


  public Connection newConn() throws SQLException {
    return DriverManager.getConnection(this.jdbcUrl, this.username, this.password);
  }


  public static void main(String[] args) throws Exception {

    MysqlCfg source = new MysqlCfg().setJdbcUrl(
            "jdbc:mysql://10.200.68.3:3306/zcwdb?zeroDateTimeBehavior=convertToNull&useSSL=false").setUsername("adm")
        .setPassword("oK1@cM2]dB2!");

    try (Connection connection = source.newConn()) {
      DatabaseMetaData metaData = connection.getMetaData();

      ResultSet primaryKeys = metaData.getPrimaryKeys(null, "zcwdb", "notify_0");
      while (primaryKeys.next()) {
        log.info("pk:{}", primaryKeys.getString("COLUMN_NAME"));
      }

      final ResultSet columns = metaData.getColumns(null, "zcwdb", "notify_0", null);
      ResultSetMetaData metaData1 = columns.getMetaData();
      int columnCount = metaData1.getColumnCount();
      for (int i = 1; i <= columnCount; i++) {
        log.info("name:{}, type:{}", metaData1.getColumnName(i), metaData1.getColumnTypeName(i));
      }

      while (columns.next()) {
        MysqlColumn col = MysqlColumn.fromResultSet(columns);
        log.info("col:{}", col);
      }
    }

  }


  public MysqlCfg setJdbcUrl(String jdbcUrl) {
    this.jdbcUrl = jdbcUrl;
    return this;
  }

  public MysqlCfg setUsername(String username) {
    this.username = username;
    return this;
  }

  public MysqlCfg setPassword(String password) {
    this.password = password;
    return this;
  }
}
