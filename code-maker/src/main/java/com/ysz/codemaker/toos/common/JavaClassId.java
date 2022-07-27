package com.ysz.codemaker.toos.common;

import lombok.Getter;
import lombok.ToString;

@ToString
@Getter
public class JavaClassId {

  private String pkgName;
  private String className;

  public JavaClassId(String pkgName, String className) {
    this.pkgName = pkgName;
    this.className = className;
  }
}
