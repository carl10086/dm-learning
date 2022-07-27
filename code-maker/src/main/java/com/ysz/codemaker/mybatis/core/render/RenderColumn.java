package com.ysz.codemaker.mybatis.core.render;

import lombok.Getter;
import lombok.ToString;

@ToString
@Getter
public class RenderColumn {

  private String javaName;
  private String colName;
  private String javaTypeName;

  public RenderColumn setJavaName(String javaName) {
    this.javaName = javaName;
    return this;
  }

  public RenderColumn setColName(String colName) {
    this.colName = colName;
    return this;
  }

  public RenderColumn setJavaTypeName(String javaTypeName) {
    this.javaTypeName = javaTypeName;
    return this;
  }
}
