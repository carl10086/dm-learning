package com.ysz.dm.netty.http;

public interface DtHttpRequest {

  /**
   * 支持的 http 版本
   */
  enum Version {
    HTTP_1_0,
    HTTP_1_1
  }

  /**
   * 支持的 http 方法
   */
  enum Method {
    GET, POST, PUT, DELETE, OPTIONS, HEAD, PATCH, TRACE, CONNECT
  }

  



}
