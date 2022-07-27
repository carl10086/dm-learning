package com.ysz.codemaker.toos.mysql.core;

import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import java.sql.ResultSet;
import java.util.Locale;
import java.util.Objects;
import lombok.Getter;
import lombok.ToString;

/**
 * 封装  mysql 的列信息
 */
@Getter
@ToString
public class MysqlColumn {

  /**
   * 字段名称
   */
  private String columnName;

  /**
   * 精确的 datatype 可以去查 java.sql.Types 感觉不一定准啊
   */
  private int dataType;

  /**
   * 类型名称
   */
  private String typeName;

  /**
   * 字段大小
   */
  private int columnSize;


  private int nullable;

  private String auto;

  private MysqlJavaTypeMapping javaTypeMapping;


  public MysqlColumn setColumnName(String columnName) {
    this.columnName = columnName;
    return this;
  }

  public MysqlColumn setDataType(int dataType) {
    this.dataType = dataType;
    this.javaTypeMapping = MysqlJavaTypeMapping.fromMysqlColType(dataType);
    Preconditions.checkNotNull(this.javaTypeMapping, "unsupport mysql data type:%s", dataType);
    return this;
  }

  public MysqlColumn setTypeName(String typeName) {
    this.typeName = typeName;
    return this;
  }

  public MysqlColumn setColumnSize(int columnSize) {
    this.columnSize = columnSize;
    return this;
  }

  public MysqlColumn setNullable(int nullable) {
    this.nullable = nullable;
    return this;
  }

  public MysqlColumn setAuto(String auto) {
    this.auto = auto;
    return this;
  }


  public static MysqlColumn fromResultSet(ResultSet rs) {
    try {
      MysqlColumn mysqlColumn = new MysqlColumn();
      mysqlColumn.setColumnName(rs.getString("COLUMN_NAME"));
      mysqlColumn.setDataType(rs.getInt("DATA_TYPE"));
      mysqlColumn.setTypeName(rs.getString("TYPE_NAME"));
      mysqlColumn.setColumnSize(rs.getInt("COLUMN_SIZE"));
      mysqlColumn.setNullable(rs.getInt("NULLABLE"));
      mysqlColumn.setAuto(rs.getString("IS_AUTOINCREMENT"));
      return mysqlColumn;
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }


  public boolean auto() {
    return Objects.equals("yes", this.auto.toLowerCase(Locale.ROOT));
  }
}
