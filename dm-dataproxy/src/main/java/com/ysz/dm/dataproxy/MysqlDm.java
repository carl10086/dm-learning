package com.ysz.dm.dataproxy;

import com.zaxxer.hikari.HikariDataSource;
import java.sql.Connection;
import java.sql.Statement;
import java.util.Map;
import java.util.Map.Entry;
import net.ttddyy.dsproxy.support.ProxyDataSource;
import net.ttddyy.dsproxy.support.ProxyDataSourceBuilder;
import org.apache.commons.dbutils.QueryRunner;
import org.apache.commons.dbutils.handlers.MapHandler;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

public class MysqlDm {

  private ProxyDataSource dataSource;
  private QueryRunner queryRunner;

  @Before
  public void setUp() throws Exception {
    dataSource = createDatasource();
    queryRunner = new QueryRunner(dataSource);
  }

  private ProxyDataSource createDatasource() throws Exception {
    /*配置优化基于官方推荐 https://github.com/brettwooldridge/HikariCP/wiki/MySQL-Configuration*/
    HikariDataSource ds = new HikariDataSource();
    ds.setDriverClassName("com.mysql.jdbc.Driver");
    ds.setAutoCommit(true);
    ds.setMaximumPoolSize(10);
    ds.setMinimumIdle(10);
    ds.setReadOnly(false);
    ds.setJdbcUrl(
        "jdbc:mysql://10.1.4.11:3306/zcwdb?useUnicode=true&characterEncoding=utf-8&zeroDateTimeBehavior=convertToNull");
    ds.setUsername("zcw_db_user");
    ds.setPassword("PZ4tEcNVrLhcPxUt");
    ds.addDataSourceProperty("cachePrepStmts", "true");
    ds.addDataSourceProperty("prepStmtCacheSize", "250");
    ds.addDataSourceProperty("prepStmtCacheSqlLimit", "2048");
    ds.addDataSourceProperty("useServerPrepStmts", "true");
    ds.addDataSourceProperty("useLocalSessionState", "true");
    ds.addDataSourceProperty("rewriteBatchedStatements", "true");
    ds.addDataSourceProperty("cacheResultSetMetadata", "true");
    ds.addDataSourceProperty("cacheServerConfiguration", "true");
    ds.addDataSourceProperty("elideSetAutoCommits", "true");
    ds.addDataSourceProperty("maintainTimeStats", "false");
    ProxyDataSource proxyDataSource = ProxyDataSourceBuilder.create(ds).afterQuery(
        (executionInfo, list) -> {
          list.forEach(x -> System.out.println(x.getQuery()));
        }).build();
    return proxyDataSource;
  }

  @Test
  public void testSelect() throws Exception {
    Map<String, Object> query = queryRunner
        .query(
            "SELECT * FROM blog_album WHERE id in ( SELECT id FROM message_message111 where  id = 512)",
            new MapHandler());
    for (Entry<String, Object> stringObjectEntry : query.entrySet()) {
      System.out.println(stringObjectEntry.getKey() + ":" + stringObjectEntry.getValue());
    }
  }


  @Test
  public void tstTimeoutException() throws Exception {
    try (Connection connection = dataSource.getConnection(); Statement statement = connection
        .createStatement()) {
      statement.setQueryTimeout(1);
      statement.execute("SELECT * FROM message_message ORDER BY sender_id desc");
    }
  }


  @After
  public void tearDown() throws Exception {
    dataSource.close();
  }
}
