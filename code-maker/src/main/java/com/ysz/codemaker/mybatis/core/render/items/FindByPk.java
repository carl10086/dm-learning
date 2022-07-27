package com.ysz.codemaker.mybatis.core.render.items;

import lombok.Getter;
import lombok.ToString;

@ToString
@Getter
public class FindByPk {

  private String parameterType;
  private String tableName;
  private String whereSql;

  public FindByPk setParameterType(String parameterType) {
    this.parameterType = parameterType;
    return this;
  }

  public FindByPk setTableName(String tableName) {
    this.tableName = tableName;
    return this;
  }

  public FindByPk setWhereSql(String whereSql) {
    this.whereSql = whereSql;
    return this;
  }
}
