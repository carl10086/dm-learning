package com.ysz.codemaker.toos.mysql.core;

import java.util.ArrayList;
import java.util.List;
import lombok.Getter;
import lombok.ToString;

@Getter
@ToString
public class MysqlMeta {

  /**
   * 主键列
   */
  private List<MysqlColumn> pks;

  /**
   * 一般列
   */
  private List<MysqlColumn> columns;

  public MysqlMeta() {
    this(16);
  }

  public MysqlMeta(int columnCnt) {
    this(1, columnCnt);
  }

  public MysqlMeta(int pkCnt, int columnCnt) {
    this.pks = new ArrayList<>(pkCnt);
    this.columns = new ArrayList<>(columnCnt);
  }

  public MysqlMeta setPks(List<MysqlColumn> pks) {
    this.pks = pks;
    return this;
  }

  public MysqlMeta setColumns(List<MysqlColumn> columns) {
    this.columns = columns;
    return this;
  }


}
