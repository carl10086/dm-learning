package com.ysz.codemaker.toos.mysql.core;

import it.unimi.dsi.fastutil.ints.IntSet;
import java.math.BigDecimal;
import java.sql.Types;
import java.util.Date;
import lombok.Getter;
import lombok.ToString;

/**
 * mysql columns 的 java type 进行映射转换
 */
@ToString
@Getter
public enum MysqlJavaTypeMapping {
  java_lang_int(Integer.class,
                IntSet.of(Types.BIT, Types.TINYINT, Types.SMALLINT, Types.INTEGER)
  ), java_lang_long(Long.class, IntSet.of(Types.BIGINT)),

  java_lang_float(Float.class, IntSet.of(Types.FLOAT)),

  java_lang_double(Double.class, IntSet.of(Types.DOUBLE)),

  java_math_decimal(BigDecimal.class, IntSet.of(Types.DECIMAL)),

  java_lang_string(String.class, IntSet.of(Types.VARCHAR, Types.CHAR, Types.LONGVARCHAR, Types.BLOB)),

  java_util_date(Date.class, IntSet.of(Types.TIME, Types.TIMESTAMP, Types.DATE)),

  ;

  private final Class<?> javaClass;
  /**
   * 参考 jdbc 规范类
   *
   * @{see java.sql.Types}
   */
  private final IntSet typesSet;


  MysqlJavaTypeMapping(Class<?> javaClass, IntSet typesSet) {
    this.javaClass = javaClass;
    this.typesSet = typesSet;
  }


  public static MysqlJavaTypeMapping fromMysqlColType(int type) {
    for (MysqlJavaTypeMapping value : MysqlJavaTypeMapping.values()) {
      if (value.getTypesSet().contains(type)) {
        return value;
      }
    }

    return null;

  }
}
