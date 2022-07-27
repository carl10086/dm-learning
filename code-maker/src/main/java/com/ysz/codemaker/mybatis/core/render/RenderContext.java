package com.ysz.codemaker.mybatis.core.render;

import com.ysz.codemaker.mybatis.core.render.items.FindByPk;
import com.ysz.codemaker.mybatis.core.render.items.InsertOne;
import com.ysz.codemaker.mybatis.core.render.items.UpdateByVersion;
import java.util.List;
import lombok.Getter;
import lombok.ToString;

@ToString
@Getter
public class RenderContext {

  private List<RenderColumn> pks;
  private List<RenderColumn> cols;
  private String allCols;

  private FindByPk findByPk;
  private UpdateByVersion updateByVersion;
  private InsertOne insertOne;

  public RenderContext setPks(List<RenderColumn> pks) {
    this.pks = pks;
    return this;
  }

  public RenderContext setCols(List<RenderColumn> cols) {
    this.cols = cols;
    return this;
  }

  public RenderContext setAllCols(String allCols) {
    this.allCols = allCols;
    return this;
  }

  public RenderContext setFindByPk(FindByPk findByPk) {
    this.findByPk = findByPk;
    return this;
  }

  public RenderContext setUpdateByVersion(UpdateByVersion updateByVersion) {
    this.updateByVersion = updateByVersion;
    return this;
  }

  public RenderContext setInsertOne(InsertOne insertOne) {
    this.insertOne = insertOne;
    return this;
  }
}
