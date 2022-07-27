package com.ysz.codemaker.mybatis;

import com.ysz.codemaker.toos.mysql.core.MysqlCfg;
import com.ysz.codemaker.mybatis.core.Cfg;
import com.ysz.codemaker.mybatis.core.Output;
import org.junit.Test;

public class MybatisCodeMakerTest {

  @Test
  public void execute() throws Exception {
    String jdbcUrl = "jdbc:mysql://10.200.68.3:3306/zcwdb?zeroDateTimeBehavior=convertToNull&useSSL=false";
    String username = "adm";
    String password = "oK1@cM2]dB2!";

    MysqlCfg mysqlCfg = new MysqlCfg()
        .setJdbcUrl(jdbcUrl)
        .setUsername(username)
        .setPassword(password);

    Cfg cfg = new Cfg().setMysql(mysqlCfg).setDatabase("zcwdb").setTableName("auth_user").setVersionColName("version");

    Output output = new MybatisCodeMaker().execute(cfg);

    System.out.println(output.getMapperXml());


  }
}