package com.ysz.codemaker.mybatis.core;

public enum DefaultMappingStrategy {
  /**
   * 什么都不做, 直接转
   */
  NOTHING,
  /**
   * 首字母小写的下划线 -> 首字母小写的驼峰
   */
  LOWER_UNDERSCORE_2_LOWER_CAMEL,

}
