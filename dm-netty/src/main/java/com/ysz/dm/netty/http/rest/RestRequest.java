package com.ysz.dm.netty.http.rest;

import java.util.Map;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

@ToString
@Setter
public class RestRequest {

  private Map<String, String> params;

  private String rawPath;

  public Map<String, String> params() {
    return params;
  }

  public String rawPath() {
    return rawPath;
  }

  public enum Method {
    GET, POST, PUT, DELETE, OPTIONS, HEAD, PATCH, TRACE, CONNECT
  }
}
