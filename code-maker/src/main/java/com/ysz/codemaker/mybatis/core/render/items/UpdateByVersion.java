package com.ysz.codemaker.mybatis.core.render.items;

import com.ysz.codemaker.mybatis.core.render.RenderColumn;
import java.util.List;
import lombok.Getter;
import lombok.ToString;

@ToString
@Getter
public class UpdateByVersion {


  private RenderColumn version;
  private List<RenderColumn> cols;
  private String tableName;
  private String whereSql;

  public UpdateByVersion setCols(List<RenderColumn> cols) {
    this.cols = cols;
    return this;
  }

  public UpdateByVersion setVersion(RenderColumn version) {
    this.version = version;
    return this;
  }

  public UpdateByVersion setTableName(String tableName) {
    this.tableName = tableName;
    return this;
  }

  public UpdateByVersion setWhereSql(String whereSql) {
    this.whereSql = whereSql;
    return this;
  }
}
