package com.ysz.codemaker.toos.mysql;

import com.ysz.codemaker.toos.mysql.core.MysqlCfg;
import com.ysz.codemaker.toos.mysql.core.MysqlMeta;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;

@Slf4j
public class MysqlMetaQueryTest {

  private MysqlMetaQuery mysqlMetaQuery = new MysqlMetaQuery();


  @Test
  public void testQueryMeta() throws Exception {
    /*云上 测试 db*/
    String jdbcUrl = "jdbc:mysql://10.200.68.3:3306/zcwdb?zeroDateTimeBehavior=convertToNull&useSSL=false";
    String username = "adm";
    String password = "oK1@cM2]dB2!";

    MysqlMeta mysqlMeta = this.mysqlMetaQuery.queryColumns(
        new MysqlCfg()
            .setJdbcUrl(jdbcUrl)
            .setUsername(username)
            .setPassword(password),
        "zcwdb",
        "notify_0"
    );

    mysqlMeta.getPks().forEach(x -> log.info("pks:{}", x));
    mysqlMeta.getColumns().forEach(x -> log.info("cols:{}", x));

  }

}