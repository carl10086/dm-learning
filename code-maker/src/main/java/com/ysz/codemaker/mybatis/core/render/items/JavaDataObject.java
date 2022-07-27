package com.ysz.codemaker.mybatis.core.render.items;

import com.ysz.codemaker.mybatis.core.render.RenderColumn;
import com.ysz.codemaker.toos.common.JavaClassId;
import java.util.List;
import lombok.Getter;
import lombok.ToString;

@ToString
@Getter
public class JavaDataObject {

  private JavaClassId id;

  private List<RenderColumn> cols;

}
