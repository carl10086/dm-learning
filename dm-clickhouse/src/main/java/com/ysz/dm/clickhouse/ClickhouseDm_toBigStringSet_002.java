package com.ysz.dm.clickhouse;

import it.unimi.dsi.fastutil.objects.ObjectOpenHashBigSet;
import java.sql.Connection;
import java.sql.DriverManager;
import java.util.Objects;
import org.apache.commons.dbutils.QueryRunner;

public class ClickhouseDm_toBigStringSet_002 {

  public static void main(String[] args) throws Exception {
    long start = System.currentTimeMillis();
    Connection connection = DriverManager.getConnection("jdbc:clickhouse://10.1.3.101:8123");
    QueryRunner queryRunner = new QueryRunner();
    ObjectOpenHashBigSet<String> query = queryRunner.query(connection,
//        "select distinct(`$device_id`)  FROM dw.pickyou_events_v1",
        "SELECT distinct(auth_user_id) FROM dw.t_nginx_www_v2 WHERE toYYYYMMDD(time_iso8601) > 20200501 ",
        resultSet -> {
          resultSet.setFetchSize(10000);
          ObjectOpenHashBigSet<String> res = new ObjectOpenHashBigSet<>();
          while (resultSet.next()) {
            res.add(Objects.toString(resultSet.getObject(1)));
          }
          return res;
        }
    );

    System.out.println(query.size64() + "-timeCost:" + (System.currentTimeMillis() - start));

    connection.close();
  }

}
