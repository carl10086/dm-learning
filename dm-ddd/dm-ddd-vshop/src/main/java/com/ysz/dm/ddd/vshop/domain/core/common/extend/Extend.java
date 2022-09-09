package com.ysz.dm.ddd.vshop.domain.core.common.extend;

import lombok.Getter;
import lombok.ToString;

/**
 * 扩展类.
 *
 * 扩展类的解析模式由 type 和 extendSchema 共同决定
 *
 * @author carl
 * @create 2022-09-09 2:43 PM
 **/
@ToString
@Getter
public class Extend {

  private ExtendType type;

  private Class<?> extendSchema;

  private String value;

}
