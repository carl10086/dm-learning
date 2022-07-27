package com.ysz.codemaker.mybatis.core.render.items;

import com.ysz.codemaker.mybatis.core.render.RenderColumn;
import com.ysz.codemaker.toos.common.JavaClassId;
import java.util.List;
import lombok.Getter;
import lombok.ToString;

@ToString
@Getter
public class InsertOne {

  private String useGeneratedKeysStr = "";

  private JavaClassId classId;

  private List<RenderColumn> cols;

  private String tableName;

  public InsertOne setUseGeneratedKeysStr(String useGeneratedKeysStr) {
    this.useGeneratedKeysStr = useGeneratedKeysStr;
    return this;
  }

  public InsertOne setClassId(JavaClassId classId) {
    this.classId = classId;
    return this;
  }

  public InsertOne setCols(List<RenderColumn> cols) {
    this.cols = cols;
    return this;
  }

  public InsertOne setTableName(String tableName) {
    this.tableName = tableName;
    return this;
  }
}
