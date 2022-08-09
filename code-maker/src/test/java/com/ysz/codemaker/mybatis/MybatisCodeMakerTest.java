package com.ysz.codemaker.mybatis;

import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import com.ysz.codemaker.mybatis.core.Cfg;
import com.ysz.codemaker.mybatis.core.Output;
import com.ysz.codemaker.toos.common.JavaClassId;
import com.ysz.codemaker.toos.mysql.core.MysqlCfg;
import java.io.File;
import org.junit.Before;
import org.junit.Test;

public class MybatisCodeMakerTest {

  private Config config;

  @Before
  public void setUp() throws Exception {
    this.config = ConfigFactory.parseFile(new File("/Users/carl/work/dt/conf/prism.properties"));
  }

  @Test
  public void execute() throws Exception {
    String jdbcUrl = config.getString("uactdb.jdbcurl");
    String username = config.getString("uactdb.username");
    String password = config.getString("uactdb.password");

    MysqlCfg mysqlCfg = new MysqlCfg().setJdbcUrl(jdbcUrl).setUsername(username).setPassword(password);

    Cfg cfg = new Cfg().setMysql(mysqlCfg).setDatabase("uactdb").setTableName("uact_user_cnt_0")
        .setDataObjectClass(new JavaClassId(
            "com.duitang.uact.srv.port.persist.dataobject",
            "UserCntDO"
        ))
        .setVersionColName("version");

    Output output = new MybatisCodeMaker().execute(cfg);

    System.out.println(output.getMapperXml());


  }
}