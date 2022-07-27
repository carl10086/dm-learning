package com.ysz.codemaker.mybatis.core;

import lombok.Getter;
import lombok.ToString;

@ToString
@Getter
public class Output {

  private String mapperXml;

  private String dataObjectJavaFile;


  public Output setMapperXml(String mapperXml) {
    this.mapperXml = mapperXml;
    return this;
  }


  public Output setDataObjectJavaFile(String dataObjectJavaFile) {
    this.dataObjectJavaFile = dataObjectJavaFile;
    return this;
  }
}
