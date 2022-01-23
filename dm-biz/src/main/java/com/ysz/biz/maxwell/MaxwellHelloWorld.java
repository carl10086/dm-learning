package com.ysz.biz.maxwell;

import com.zendesk.maxwell.Maxwell;
import com.zendesk.maxwell.MaxwellConfig;
import com.zendesk.maxwell.filtering.Filter;
import com.zendesk.maxwell.filtering.InvalidFilterException;
import com.zendesk.maxwell.replication.BinlogPosition;
import com.zendesk.maxwell.replication.Position;

public class MaxwellHelloWorld {

  public void tst() throws Exception {
    final MaxwellConfig config = mockCfg();
    final Maxwell maxwell = new Maxwell(config);
    maxwell.start();
  }

  public static void main(String[] args) throws Exception {
    new MaxwellHelloWorld().tst();
  }


  private MaxwellConfig mockCfg() throws InvalidFilterException {
//    String cmdLine = "--user='adm' --password='123456' --host='10.200.68.3' --producer=stdout";
//    return new MaxwellConfig(new String[]{cmdLine});
    MaxwellConfig config = new MaxwellConfig();
    config.filter = new Filter();
    config.filter.addRule("include: carl.*");
//    config.filter.addRule("exclude: carl.*");
    config.maxwellMysql.user = "adm";
    config.maxwellMysql.password = "123456";
    config.maxwellMysql.host = "10.200.68.3";

    config.initPosition = new Position(
        new BinlogPosition(98239L, "68003-bin.000003"), 0L
    );
    config.replicationMysql = config.maxwellMysql;
    config.producerType = "stdout";

    config.outputConfig.includesBinlogPosition = true;
    config.outputConfig.includesXOffset = true;
    return config;
  }

}
