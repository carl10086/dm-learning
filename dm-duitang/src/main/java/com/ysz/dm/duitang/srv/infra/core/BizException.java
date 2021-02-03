package com.ysz.dm.duitang.srv.infra.core;

public class BizException extends RuntimeException {

  public BizException() {
  }

  public BizException(final String message) {
    super(message);
  }
}
