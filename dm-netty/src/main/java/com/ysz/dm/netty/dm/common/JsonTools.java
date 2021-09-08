package com.ysz.dm.netty.dm.common;

import com.fasterxml.jackson.databind.JavaType;

/** @author carl */
public class JsonTools {

  public static JsonMapper getMapper() {
    return SingletonHolder.jsonMapper;
  }

  public static String toJson(Object object) {
    if (object == null) {
      return null;
    }
    return getMapper().toJson(object);
  }

  public static <T> T fromJson(String jsonString, Class<T> clazz) {
    if (jsonString == null) {
      return null;
    }
    return getMapper().fromJson(jsonString, clazz);
  }

  public static <T> T fromJson(String jsonString, JavaType javaType) {
    if (jsonString == null) {
      return null;
    }
    return getMapper().fromJson(jsonString, javaType);
  }

  private static class SingletonHolder {

    static JsonMapper jsonMapper = JsonMapper.defaultMapper();
  }
}
