package com.ysz.dm.clickhouse;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.Statement;
import java.util.Arrays;

public class ClickhouseHttpDm {

  public static void main(String[] args) throws Exception {
    Connection connection = DriverManager.getConnection("jdbc:clickhouse://10.1.13.54:9000");

    Statement statement = connection.createStatement();
//    statement.addBatch(
//        "INSERT INTO dw.users_v2 (uid, gender, createAt, status, latitude, longitude, minAge, maxAge, updateAt, userTagStatus)");
//    statement.addBatch(
//        "alter table dw.users_v2 update gender = 4, minAge = 10, status = 'fdasfds', updateAt = now() where toYYYYMMDD(createAt) = 20200514 AND uid = 111111111111;");
//    statement.executeBatch();

    statement.addBatch(
        "INSERT INTO dw.users_v2 (uid, gender, createAt, status, latitude, longitude, minAge, maxAge, updateAt, userTagStatus) VALUES (111111111111, 1, now(), '1', 1.0, 1.0, 1, 100, now(), 'chosen')");
//    statement.addBatch(
//        "alter table dw.users_v2 update gender = 4, minAge = 10, status = 'fdasfds', updateAt = now() where toYYYYMMDD(createAt) = 20200514 AND uid = 111111111111;");
    int[] ints = statement.executeBatch();
    System.out.println(Arrays.toString(ints));
    connection.close();
  }

}
