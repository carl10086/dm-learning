package com.ysz.codemaker.mybatis;

import com.ysz.codemaker.toos.mysql.core.MysqlCfg;
import com.ysz.codemaker.mybatis.core.Cfg;
import com.ysz.codemaker.mybatis.core.Output;
import org.junit.Test;

public class MybatisCodeMakerTest {

  @Test
  public void execute() throws Exception {
    String jdbcUrl = "jdbc:mysql://10.200.68.3:3306/uactdb?zeroDateTimeBehavior=convertToNull&useSSL=false";
    String username = "adm";
    String password = "123456";

    MysqlCfg mysqlCfg = new MysqlCfg()
        .setJdbcUrl(jdbcUrl)
        .setUsername(username)
        .setPassword(password);

    Cfg cfg = new Cfg().setMysql(mysqlCfg).setDatabase("uactdb").setTableName("uact_likes_0")
        .setVersionColName("version");

    Output output = new MybatisCodeMaker().execute(cfg);

    System.out.println(output.getMapperXml());


  }
}