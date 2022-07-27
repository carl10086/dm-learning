package com.ysz.codemaker.toos.mysql;

import com.ysz.codemaker.toos.mysql.core.MysqlCfg;
import com.ysz.codemaker.toos.mysql.core.MysqlColumn;
import com.ysz.codemaker.toos.mysql.core.MysqlMeta;
import java.sql.Connection;
import java.sql.DatabaseMetaData;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import java.util.HashSet;
import java.util.Set;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public class MysqlMetaQuery {


  public MysqlMeta queryColumns(
      MysqlCfg cfg, String database, String table
  ) throws SQLException {

    try (Connection connection = cfg.newConn()) {
      DatabaseMetaData metaData = connection.getMetaData();

      /*1. query pk names*/
      Set<String> pks = queryPks(database, table, metaData);

      /*2. query column meta*/
      final ResultSet columns = metaData.getColumns(null, database, table, null);
      debug(columns);

      MysqlMeta mysqlMeta = new MysqlMeta();
      while (columns.next()) {
        MysqlColumn col = MysqlColumn.fromResultSet(columns);
        if (pks.contains(col.getColumnName())) {
          mysqlMeta.getPks().add(col);
        } else {
          mysqlMeta.getColumns().add(col);
        }
      }

      return mysqlMeta;
    }

  }

  private void debug(ResultSet columns) throws SQLException {
    ResultSetMetaData metaData1 = columns.getMetaData();
    int columnCount = metaData1.getColumnCount();
    for (int i = 1; i <= columnCount; i++) {
      log.debug("name:{}, type:{}", metaData1.getColumnName(i), metaData1.getColumnTypeName(i));
    }
  }

  private Set<String> queryPks(String database, String table, DatabaseMetaData metaData) throws SQLException {
    ResultSet primaryKeys = metaData.getPrimaryKeys(null, database, table);

    Set<String> pks = new HashSet<>();
    while (primaryKeys.next()) {
      pks.add(primaryKeys.getString("COLUMN_NAME"));
    }
    return pks;
  }

}
