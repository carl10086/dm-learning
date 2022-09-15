package com.ysz.dm.ddd.vshop.domain.core.common.lang;

import lombok.Getter;
import lombok.ToString;

/**
 * @author carl
 * @create 2022-09-15 5:21 PM
 **/
@ToString
@Getter
public final class JsonString<V> {

  private String jsonContent;

  private Class<V> clz;

  public JsonString<V> setJsonContent(String jsonContent) {
    this.jsonContent = jsonContent;
    return this;
  }

  public JsonString<V> setClz(Class<V> clz) {
    this.clz = clz;
    return this;
  }
}
